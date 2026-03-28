"""Layer path construction and model relayering for RYS (i,j) configs.

The core trick: for each config we temporarily swap the model's nn.ModuleList
with a shallow-copied sequence of layer objects that follows the (i,j) path.
Using copy.copy() (not direct references) is critical — it creates a new
Python object with independent _modules / _parameters dicts while still sharing
all weight tensors.  Without this, duplicated layers share internal state and
corrupt the KV cache when the same layer appears twice in the path.

We also reassign the `layer_idx` attribute on each layer's self-attention
module so that RoPE / KV cache indexing uses the *position in the path* rather
than the original layer number.  Without this, two copies of layer 17 would
both try to write to KV cache slot 17, causing silent correctness errors.

Architecture detection note
----------------------------
Most HuggingFace decoder-only models store transformer layers at:
    model.model.layers   ← Qwen2, Llama 2/3, Mistral, Gemma, Phi-3, etc.

A smaller set of multimodal or non-standard models use a different attribute
path, e.g.:
    model.language_model.model.layers   ← LLaVA-style
    model.transformer.h                 ← GPT-2 / Falcon (old-style)

For now we hardcode the `model.model.layers` path because it covers the vast
majority of modern open-weights instruction-tuned models that RYS targets.
If you hit an AttributeError here, add a detection branch below rather than
changing the hardcoded path.  The RYS repo (dnhkng/RYS src/core/) handles
MoE variants via a separate layer_duplicator_moe.py — that is not implemented
here yet.

TODO: implement auto-detection via _find_layers() for GPT-2 / multimodal models.
TODO: implement MoE support (see RYS src/core/layer_duplicator_moe.py).
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from typing import Any


# ---------------------------------------------------------------------------
# Layer path construction
# ---------------------------------------------------------------------------

def build_layer_path(i: int, j: int, num_layers: int) -> list[int]:
    """Return the ordered list of layer indices for config (i, j).

    Baseline (0, 0) → straight pass through all layers.
    Any other (i, j) → [0..j-1] + [i..N-1], so layers i..j-1 run twice.

    Example for (2, 7) with N=9:
        first_pass  = [0, 1, 2, 3, 4, 5, 6]
        second_pass = [2, 3, 4, 5, 6, 7, 8]
        result      = [0, 1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 7, 8]
    """
    if i == 0 and j == 0:
        return list(range(num_layers))
    first_pass = list(range(0, j))
    second_pass = list(range(i, num_layers))
    return first_pass + second_pass


def get_duplicated_layers(i: int, j: int) -> list[int]:
    """Return the indices of layers that are traversed twice."""
    if i == 0 and j == 0:
        return []
    return list(range(i, j))


def generate_all_configs(num_layers: int) -> list[tuple[int, int]]:
    """Generate the full (i, j) sweep queue for an N-layer model.

    Matches the RYS repo's scripts/init_queue.py.
    Total configs = N*(N+1)/2 + 1 (baseline + all valid pairs).

    Note: RYS iterates j-outer, i-inner (column-wise).  We iterate i-outer,
    j-inner (row-wise) here — the total set of configs is identical, only the
    evaluation order differs, which doesn't affect correctness.
    """
    configs: list[tuple[int, int]] = [(0, 0)]  # baseline first
    for i in range(num_layers):
        for j in range(i + 1, num_layers + 1):
            configs.append((i, j))
    return configs


# ---------------------------------------------------------------------------
# Model relayering
# ---------------------------------------------------------------------------

def _get_layers(model: Any) -> nn.ModuleList:
    """Return the model's transformer layer list.

    Hardcoded to model.model.layers — covers Qwen2, Llama, Mistral, Gemma,
    Phi-3, and most modern HF decoder-only models.

    If this raises AttributeError for your model, check:
      - model.language_model.model.layers  (LLaVA / multimodal variants)
      - model.transformer.h                (GPT-2 / old Falcon style)
    Add a detection branch here rather than patching the call sites.
    """
    return model.model.layers


def _set_layers(model: Any, layers: nn.ModuleList) -> None:
    """Assign a new ModuleList to the model's layer slot."""
    model.model.layers = layers


def _set_layer_idx(layer: Any, idx: int) -> None:
    """Set the layer_idx on the layer's self-attention module (if present).

    This is required for correct KV cache indexing when the same layer object
    appears multiple times in the execution path.  Without this, duplicate
    layers share the same cache slot and corrupt each other's state.

    Tries common attribute names for the self-attention sub-module across
    different model families.
    """
    # Common self-attention attribute names across HF model families
    attn_attrs = ("self_attn", "attention", "attn", "self_attention")
    for attr in attn_attrs:
        attn = getattr(layer, attr, None)
        if attn is not None and hasattr(attn, "layer_idx"):
            attn.layer_idx = idx
            return
    # If no self_attn with layer_idx found, silently skip.
    # Not all models use layer_idx for KV cache management.


def relayer_model(
    model: Any,
    i: int,
    j: int,
    num_layers: int,
) -> tuple[nn.ModuleList, dict[str, Any]]:
    """Temporarily rewire the model to follow the (i,j) execution path.

    Uses shallow copies (copy.copy) of layer objects so that:
    - Weight tensors are shared (no VRAM increase)
    - Internal module dicts are independent (no cross-contamination)
    - layer_idx can be safely reassigned per position in the path

    Returns (original_layers, original_config_state) so the caller can
    restore the model after inference.  Always call restore_model() after this,
    even if inference raises — use a try/finally block.

    Patches the following model.config attributes to match the new path length:
    - num_hidden_layers  (always)
    - layer_types        (Qwen2 4.50+ sliding-window attention type per layer)
                         Without this patch the Qwen2 forward pass does
                         `self.config.layer_types[i]` where i can exceed the
                         original list length, causing IndexError on every
                         non-baseline config.  We remap by original layer index
                         so duplicate layers get the correct attention type.

    If a future model adds other per-layer config lists, add them here.
    """
    original_layers = _get_layers(model)

    # Snapshot every config attribute we will mutate so restore_model can put
    # them back exactly.  Use a plain dict — no magic.
    original_config_state: dict[str, Any] = {
        "num_hidden_layers": model.config.num_hidden_layers,
    }

    # layer_types: added in transformers 4.50 for Qwen2 sliding-window support.
    # It is a list of strings (e.g. ["full_attention", "sliding_window", ...])
    # with one entry per original layer.  We MUST remap it along with the path
    # or the Qwen2 forward pass raises IndexError for any config where
    # len(path) > num_layers.
    layer_types = getattr(model.config, "layer_types", None)
    if layer_types is not None:
        original_config_state["layer_types"] = list(layer_types)

    path = build_layer_path(i, j, num_layers)

    new_layer_list: list[Any] = []
    for path_pos, layer_idx in enumerate(path):
        layer_copy = copy.copy(original_layers[layer_idx])
        _set_layer_idx(layer_copy, path_pos)
        new_layer_list.append(layer_copy)

    _set_layers(model, nn.ModuleList(new_layer_list))
    model.config.num_hidden_layers = len(path)

    if layer_types is not None:
        model.config.layer_types = [layer_types[idx] for idx in path]

    return original_layers, original_config_state


def restore_model(
    model: Any,
    original_layers: nn.ModuleList,
    original_config_state: dict[str, Any],
) -> None:
    """Restore the model to its original layer configuration.

    Restores all config attributes saved by relayer_model, then reinstates
    the original ModuleList and resets layer_idx on each original layer object.
    """
    # Restore config attrs first (num_hidden_layers, layer_types, ...)
    for attr, val in original_config_state.items():
        setattr(model.config, attr, val)

    # Restore original layer_idx values on the original layer objects
    for idx, layer in enumerate(original_layers):
        _set_layer_idx(layer, idx)

    _set_layers(model, original_layers)
