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

Added 4/4/26: implement auto-detection via get_num_layers to probe for model type and determine the layer path.
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
# Config helpers
# ---------------------------------------------------------------------------

# Attribute names for the layer count across known HF config classes.
# Ordered by prevalence so the first hit is almost always right.
_LAYER_COUNT_ATTRS = (
    "num_hidden_layers",   # Llama, Qwen2, Mistral, Gemma, Phi-3, most modern
    "n_layer",             # GPT-2, BERT, RoBERTa, some Falcon variants
    "num_layers",          # T5, mT5, some encoder-decoder models
    "n_layers",            # older GPT-J / Bloom variants
    "num_decoder_layers",  # encoder-decoder fallback
)


def get_num_layers(config: Any) -> int:
    """Extract num_hidden_layers from a model config, trying common attr names.

    Raises ValueError if none of the known attribute names are present.
    Also handles multimodal configs where the value is nested under text_config.
    """
    for attr in _LAYER_COUNT_ATTRS:
        val = getattr(config, attr, None)
        if val is not None:
            return int(val)

    # Multimodal / vision-language models often nest the LLM config
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is not None:
        for attr in _LAYER_COUNT_ATTRS:
            val = getattr(text_cfg, attr, None)
            if val is not None:
                return int(val)

    raise ValueError(
        f"Cannot determine layer count from {type(config).__name__}. "
        f"Checked attributes: {_LAYER_COUNT_ATTRS}. "
        "Please open a GitHub issue or pass --num-layers manually."
    )


# ---------------------------------------------------------------------------
# Model relayering
# ---------------------------------------------------------------------------

# Ordered candidate paths to the transformer layer list in the loaded model.
# Each entry is a dotted attribute path.  The first one that resolves wins.
_LAYER_PATHS = [
    "model.layers",                     # Qwen2, Llama 2/3, Mistral, Gemma, Phi-3
    "model.language_model.model.layers",# Mistral3ForConditionalGeneration (VLM)
    "language_model.model.layers",      # LLaVA-style multimodal
    "model.model.layers",               # some double-wrapped configs
    "transformer.h",                    # GPT-2 / old Falcon
    "transformer.blocks",               # MPT
]


def _resolve_attr_path(obj: Any, path: str) -> Any | None:
    """Walk a dotted attribute path, returning None if any step is missing."""
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def _detect_layer_path(model: Any) -> str:
    """Return the dotted path to the model's transformer layer ModuleList.

    Tries _LAYER_PATHS in order and returns the first one that resolves to a
    non-empty nn.ModuleList.  Raises ValueError if none match.
    """
    for path in _LAYER_PATHS:
        result = _resolve_attr_path(model, path)
        if isinstance(result, nn.ModuleList) and len(result) > 0:
            return path

    raise ValueError(
        f"Cannot find transformer layers in {type(model).__name__}. "
        f"Tried: {_LAYER_PATHS}. "
        "Please open a GitHub issue with your model architecture."
    )


def _get_layers(model: Any, layer_path: str) -> nn.ModuleList:
    """Return the model's transformer layer list at the given dotted path."""
    result = _resolve_attr_path(model, layer_path)
    if result is None:
        raise AttributeError(f"Path {layer_path!r} not found on model.")
    return result


def _set_layers(model: Any, layer_path: str, layers: nn.ModuleList) -> None:
    """Assign a new ModuleList to the model's layer slot at the given path."""
    parts = layer_path.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], layers)


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

    Returns (original_layers, state) where state is an opaque dict that must
    be passed back to restore_model().  Always call restore_model() after this,
    even if inference raises — use a try/finally block.

    Patches the following model.config attributes to match the new path length:
    - The layer count attribute (num_hidden_layers / n_layer / num_layers / ...)
    - layer_types (Qwen2 4.50+ sliding-window attention type per layer)
                  Without this patch the Qwen2 forward pass does
                  `self.config.layer_types[i]` where i can exceed the
                  original list length, causing IndexError on every
                  non-baseline config.  We remap by original layer index
                  so duplicate layers get the correct attention type.

    If a future model adds other per-layer config lists, add them here.
    """
    # Auto-detect where the layers live in this model
    layer_path = _detect_layer_path(model)
    original_layers = _get_layers(model, layer_path)

    # Find which config holds the layer count (top-level or nested text_config).
    # VLMs (e.g. Mistral3ForConditionalGeneration) store the LLM config under
    # model.config.text_config rather than at the top level.
    layer_count_cfg = model.config
    layer_count_attr = None
    for attr in _LAYER_COUNT_ATTRS:
        if getattr(model.config, attr, None) is not None:
            layer_count_attr = attr
            break
    if layer_count_attr is None:
        text_cfg = getattr(model.config, "text_config", None)
        if text_cfg is not None:
            for attr in _LAYER_COUNT_ATTRS:
                if getattr(text_cfg, attr, None) is not None:
                    layer_count_cfg = text_cfg
                    layer_count_attr = attr
                    break

    # State dict: underscore-prefixed keys are internal bookkeeping,
    # all other keys are (config_obj, original_value) pairs to restore.
    state: dict[str, Any] = {
        "_layer_path": layer_path,
        "_patches": [],  # list of (config_obj, attr, original_value)
    }
    if layer_count_attr is not None:
        state["_patches"].append(
            (layer_count_cfg, layer_count_attr, getattr(layer_count_cfg, layer_count_attr))
        )

    # layer_types: Qwen2 4.50+ per-layer attention type list — must be remapped
    layer_types = getattr(model.config, "layer_types", None)
    if layer_types is not None:
        state["_patches"].append(
            (model.config, "layer_types", list(layer_types))
        )

    path = build_layer_path(i, j, num_layers)

    new_layer_list: list[Any] = []
    for path_pos, layer_idx in enumerate(path):
        layer_copy = copy.copy(original_layers[layer_idx])
        _set_layer_idx(layer_copy, path_pos)
        new_layer_list.append(layer_copy)

    _set_layers(model, layer_path, nn.ModuleList(new_layer_list))

    # Apply config patches for the new path length
    for cfg_obj, attr, _original_val in state["_patches"]:
        if attr == "layer_types":
            # Remap per-position using the original layer index for each path slot
            setattr(cfg_obj, attr, [_original_val[idx] for idx in path])
        else:
            # Layer count attribute — set to new path length
            setattr(cfg_obj, attr, len(path))

    return original_layers, state


def restore_model(
    model: Any,
    original_layers: nn.ModuleList,
    state: dict[str, Any],
) -> None:
    """Restore the model to its original layer configuration.

    Restores all patched config attributes, reinstates the original ModuleList,
    and resets layer_idx on each original layer object.

    state must be the dict returned by relayer_model — do not modify it.
    """
    layer_path = state["_layer_path"]

    # Restore all patched config attributes
    for cfg_obj, attr, original_val in state["_patches"]:
        setattr(cfg_obj, attr, original_val)

    # Restore original layer_idx values on the original layer objects
    for idx, layer in enumerate(original_layers):
        _set_layer_idx(layer, idx)

    _set_layers(model, layer_path, original_layers)
