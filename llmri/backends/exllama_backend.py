"""ExLlamaV2/V3 backend — CUDA only, requires manual ExLlama installation.

This backend is NOT implemented yet.  It is stubbed here so the CLI can
raise a clear error rather than an ImportError if a user passes --backend exllama.

Why ExLlama matters
-------------------
ExLlamaV2/V3 can run quantised (Q4/Q8) models at 2-3x the throughput of the HF
transformers backend on NVIDIA hardware.  For a 667-config sweep on a large
model (e.g. 70B), this is the difference between a 10-hour run and a 3-hour run.

Implementation notes (for the future implementer)
--------------------------------------------------
The ExLlama approach differs from the HF backend in one important way: ExLlama
compiles a CUDA kernel for the model's layer structure at load time.  This means
we cannot swap nn.ModuleList objects at runtime — the compiled kernel doesn't
support dynamic layer re-routing.

Instead, ExLlama exposes a lower-level "custom layer execution" hook (similar to
the RYS repo's scripts/run_exllama_math_eq_combined_worker.py) that accepts a
list of layer indices and executes them in the requested order.

To implement this backend:
1. Follow the RYS repo's scripts/run_exllama_math_eq_combined_worker.py as the
   reference implementation.
2. Load the model with ExLlamaV2ModelConfig / ExLlamaV2Model.
3. For each (i,j) config, pass the build_layer_path(i, j, num_layers) sequence
   directly to ExLlama's execution engine rather than modifying a ModuleList.
4. The scoring logic (pubmedqa_scorer, eq_scorer) can be reused unchanged since
   they only consume raw text strings.

References:
  - https://github.com/turboderp/exllamav2
  - https://github.com/dnhkng/RYS/blob/main/scripts/run_exllama_math_eq_combined_worker.py
"""

from __future__ import annotations


def load_model(model_path: str, device: str):  # type: ignore[return]
    raise NotImplementedError(
        "The ExLlama backend is not yet implemented.  "
        "Use --backend hf for now.  "
        "See llmri/backends/exllama_backend.py for implementation notes."
    )


def evaluate_config(*args, **kwargs):  # type: ignore[return]
    raise NotImplementedError(
        "The ExLlama backend is not yet implemented.  "
        "Use --backend hf for now."
    )
