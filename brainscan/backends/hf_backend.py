"""HuggingFace Transformers backend.

Works on Mac (MPS), CPU, and CUDA.  Loads the model once and re-uses it across
all (i,j) configs by temporarily relayering via brainscan.relayer.

Generation is intentionally minimal — we only need 1-16 new tokens per probe:
  - PubMedQA answers: 1-3 tokens ("yes", "no", "maybe")
  - EQ-Bench answers: multi-line but still short (< 100 tokens)

We do NOT use pipeline() because we need direct access to model.model.layers
for relayering.
"""

from __future__ import annotations

import logging
import torch
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from brainscan.relayer import relayer_model, restore_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def detect_device() -> str:
    """Return the best available device string: 'cuda', 'mps', or 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_path: str,
    device: str,
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> tuple[Any, Any]:
    """Load model and tokenizer.  Returns (model, tokenizer).

    Uses float16 on CUDA/MPS, float32 on CPU (bf16 would also work on
    supported CUDA devices — but float16 is the safe default).

    cache_dir: if set, weights are stored/read from this directory instead of
        the default HF hub cache (~/.cache/huggingface/hub/).
    local_files_only: if True, never hit the network — use only what is
        already in cache_dir (or the default HF cache).  Raises an error if
        the model is not cached yet.
    """
    logger.info(f"Loading model from {model_path!r} onto {device} ...")
    if cache_dir:
        logger.info(f"Cache dir: {cache_dir}")
    if local_files_only:
        logger.info("Offline mode: using only local cached files.")

    dtype = torch.float32 if device == "cpu" else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(
        f"Loaded {model.config.model_type} with "
        f"{model.config.num_hidden_layers} layers."
    )
    return model, tokenizer


def load_model_config(
    model_path: str,
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> Any:
    """Load only the model config (no weights) to inspect architecture."""
    return AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_responses(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
    device: str,
) -> list[str]:
    """Run generation for a batch of prompts and return the new-token text.

    Prompts are processed one at a time to avoid OOM on large models or long
    EQ-Bench prompts.  Batching across prompts would speed this up on CUDA but
    adds complexity around padding / attention masks.

    Returns only the newly generated tokens (not the prompt echo).
    """
    responses: list[str] = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,       # greedy — deterministic and fast
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Decode only the newly generated tokens
        new_tokens = outputs[0][input_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        responses.append(text)

    return responses


# ---------------------------------------------------------------------------
# Per-config evaluation
# ---------------------------------------------------------------------------

def evaluate_config(
    model: Any,
    tokenizer: Any,
    i: int,
    j: int,
    num_layers: int,
    pubmedqa_probes: list[dict],
    eq_probes: list[dict],
    max_new_tokens: int,
    device: str,
    active_probes: set[str],
) -> dict[str, float]:
    """Relayer the model for (i,j), run probes, restore, return scores.

    active_probes controls which probe sets to run: {"pubmedqa", "eq"}.
    Returns a dict with keys "pubmedqa_score" and/or "eq_score".
    """
    from brainscan.scoring.pubmedqa_scorer import score_pubmedqa_batch
    from brainscan.scoring.eq_scorer import score_eq_batch

    # Relayer — returns (layers, config_state_dict) now
    original_layers, original_config_state = relayer_model(model, i, j, num_layers)

    try:
        scores: dict[str, float] = {}

        if "pubmedqa" in active_probes and pubmedqa_probes:
            prompts = [p["prompt"] for p in pubmedqa_probes]
            responses = generate_responses(
                model, tokenizer, prompts, max_new_tokens, device
            )
            scores["pubmedqa_score"] = score_pubmedqa_batch(responses, pubmedqa_probes)

        if "eq" in active_probes and eq_probes:
            # EQ-Bench needs more tokens for its structured output
            eq_max_tokens = max(max_new_tokens, 128)
            prompts = [p["prompt"] for p in eq_probes]
            responses = generate_responses(
                model, tokenizer, prompts, eq_max_tokens, device
            )
            scores["eq_score"] = score_eq_batch(responses, eq_probes)

    finally:
        # Always restore, even if generation throws
        restore_model(model, original_layers, original_config_state)

    return scores
