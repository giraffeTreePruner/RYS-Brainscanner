"""Main sweep orchestrator.

Coordinates model loading, config generation, per-config evaluation,
incremental checkpointing, and final output assembly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from llmri.relayer import build_layer_path, get_duplicated_layers, generate_all_configs, get_num_layers
from llmri.utils import (
    utc_now_iso,
    save_checkpoint,
    load_checkpoint,
    get_completed_configs,
    make_progress_bar,
    count_params,
    compute_rankings,
    build_heatmap_matrices,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_scan(
    model_path: str,
    output_path: str,
    backend: str,
    device: str,
    probes: set[str],
    pubmedqa_dataset_path: str,
    eq_dataset_path: str,
    resume: bool,
    checkpoint_every: int,
    max_new_tokens: int,
    cache_dir: str | None = None,
    offline: bool = False,
    verbose: bool = False,
) -> None:
    """Run the full (i,j) sweep and write results to output_path.

    This is the function called by the CLI after argument validation.
    It is intentionally kept free of Click dependencies so it can be called
    programmatically from other tools.
    """
    # -----------------------------------------------------------------
    # 1. Load datasets
    # -----------------------------------------------------------------
    from llmri.scoring.pubmedqa_scorer import load_pubmedqa_dataset
    from llmri.scoring.eq_scorer import load_eq_dataset

    pubmedqa_probes: list[dict] = []
    eq_probes: list[dict] = []

    if "pubmedqa" in probes:
        pubmedqa_probes = load_pubmedqa_dataset(pubmedqa_dataset_path)
        logger.info(f"Loaded {len(pubmedqa_probes)} PubMedQA probes from {pubmedqa_dataset_path}")

    if "eq" in probes:
        eq_probes = load_eq_dataset(eq_dataset_path)
        logger.info(f"Loaded {len(eq_probes)} EQ probes from {eq_dataset_path}")

    # -----------------------------------------------------------------
    # 2. Load model config (weights not needed yet for architecture info)
    # -----------------------------------------------------------------
    if backend == "hf":
        from llmri.backends.hf_backend import load_model_config, load_model, detect_device
        if device == "auto":
            device = detect_device()
        model_cfg = load_model_config(
            model_path, cache_dir=cache_dir, local_files_only=offline
        )
    elif backend == "exllama":
        from llmri.backends.exllama_backend import load_model  # type: ignore
        raise SystemExit("ExLlama backend not yet implemented. Use --backend hf.")
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    num_layers: int = get_num_layers(model_cfg)
    logger.info(f"Model has {num_layers} layers. Generating configs ...")

    # -----------------------------------------------------------------
    # 3. Generate sweep queue
    # -----------------------------------------------------------------
    all_configs = generate_all_configs(num_layers)
    total_configs = len(all_configs)
    logger.info(f"Total configs to sweep: {total_configs}")

    # -----------------------------------------------------------------
    # 4. Resume: load existing results and skip completed configs
    # -----------------------------------------------------------------
    existing_results: list[dict[str, Any]] = []
    completed: set[tuple[int, int]] = set()

    if resume:
        checkpoint = load_checkpoint(output_path)
        if checkpoint:
            existing_results = checkpoint.get("results", [])
            completed = get_completed_configs(checkpoint)
            logger.info(
                f"Resuming scan: {len(completed)} configs already done, "
                f"{total_configs - len(completed)} remaining."
            )
        else:
            logger.info("No existing checkpoint found; starting fresh.")

    pending_configs = [c for c in all_configs if c not in completed]

    if not pending_configs:
        logger.info("All configs already completed. Nothing to do.")
        return

    # -----------------------------------------------------------------
    # 5. Load model weights
    # -----------------------------------------------------------------
    model, tokenizer = load_model(
        model_path, device, cache_dir=cache_dir, local_files_only=offline
    )
    total_params_str = count_params(model)
    logger.info(f"Model loaded. Parameters: {total_params_str}")

    # -----------------------------------------------------------------
    # 6. Run baseline (0,0) — needed to compute deltas
    # -----------------------------------------------------------------
    from llmri.backends.hf_backend import evaluate_config

    baseline_scores: dict[str, float] | None = None

    # Check if baseline already exists in prior results
    for r in existing_results:
        if r["config"] == [0, 0]:
            baseline_scores = {
                "pubmedqa_score": r["pubmedqa_score"],
                "eq_score": r["eq_score"],
            }
            logger.info(f"Baseline loaded from checkpoint: {baseline_scores}")
            break

    if baseline_scores is None:
        logger.info("Evaluating baseline (0, 0) ...")
        baseline_scores = evaluate_config(
            model, tokenizer,
            i=0, j=0,
            num_layers=num_layers,
            pubmedqa_probes=pubmedqa_probes,
            eq_probes=eq_probes,
            max_new_tokens=max_new_tokens,
            device=device,
            active_probes=probes,
        )
        # Fill in any missing probe score with 0.0 (in case only one probe set is active)
        baseline_scores.setdefault("pubmedqa_score", 0.0)
        baseline_scores.setdefault("eq_score", 0.0)

    baseline_pubmedqa = baseline_scores["pubmedqa_score"]
    baseline_eq = baseline_scores["eq_score"]
    baseline_combined = (baseline_pubmedqa + baseline_eq) / 2.0

    if verbose:
        logger.info(
            f"Baseline: pubmedqa={baseline_pubmedqa:.4f} "
            f"eq={baseline_eq:.4f} combined={baseline_combined:.4f}"
        )

    # -----------------------------------------------------------------
    # 7. Prepare metadata skeleton (written to every checkpoint)
    # -----------------------------------------------------------------
    scan_start = utc_now_iso()

    metadata_base = {
        "model_name": model_path,
        "model_type": model_cfg.model_type,
        "num_layers": num_layers,
        "hidden_size": getattr(model_cfg, "hidden_size", None),
        "num_attention_heads": getattr(model_cfg, "num_attention_heads", None),
        "num_key_value_heads": getattr(model_cfg, "num_key_value_heads", None),
        "total_params_base": total_params_str,
        "backend": backend,
        "device": device,
        "scan_start_utc": scan_start,
        "scan_end_utc": None,
        "scan_duration_seconds": None,
        "total_configs": total_configs,
        "completed_configs": len(completed),
        "pubmedqa_dataset": str(pubmedqa_dataset_path),
        "pubmedqa_dataset_size": len(pubmedqa_probes),
        "eq_dataset": str(eq_dataset_path),
        "eq_dataset_size": len(eq_probes),
        "max_new_tokens": max_new_tokens,
    }

    baseline_entry = {
        "config": [0, 0],
        "pubmedqa_score": baseline_pubmedqa,
        "eq_score": baseline_eq,
        "combined_score": baseline_combined,
    }

    # -----------------------------------------------------------------
    # 8. Sweep loop
    # -----------------------------------------------------------------
    import time
    sweep_start_time = time.monotonic()

    all_results: list[dict[str, Any]] = list(existing_results)
    done_count = len(completed)

    pbar = make_progress_bar(total=total_configs, desc="LL-MRI")
    pbar.update(done_count)

    for cfg_idx, (i, j) in enumerate(pending_configs):
        scores = evaluate_config(
            model, tokenizer,
            i=i, j=j,
            num_layers=num_layers,
            pubmedqa_probes=pubmedqa_probes,
            eq_probes=eq_probes,
            max_new_tokens=max_new_tokens,
            device=device,
            active_probes=probes,
        )

        pmqa_score = scores.get("pubmedqa_score", 0.0)
        eq_score = scores.get("eq_score", 0.0)
        combined_score = (pmqa_score + eq_score) / 2.0

        layer_path = build_layer_path(i, j, num_layers)
        dup_layers = get_duplicated_layers(i, j)
        num_dup = len(dup_layers)
        param_increase_pct = round(num_dup / num_layers * 100, 2)

        result = {
            "config": [i, j],
            "pubmedqa_score": pmqa_score,
            "eq_score": eq_score,
            "combined_score": combined_score,
            "pubmedqa_delta": round(pmqa_score - baseline_pubmedqa, 6),
            "eq_delta": round(eq_score - baseline_eq, 6),
            "combined_delta": round(combined_score - baseline_combined, 6),
            "duplicated_layers": dup_layers,
            "num_duplicated": num_dup,
            "layer_path": layer_path,
            "total_layers_in_path": len(layer_path),
            "param_increase_pct": param_increase_pct,
        }
        all_results.append(result)
        done_count += 1
        pbar.update(1)

        if verbose:
            logger.debug(
                f"({i:3d},{j:3d}) pmqa={pmqa_score:.4f} "
                f"eq={eq_score:.4f} combined={combined_score:.4f} "
                f"Δcombined={result['combined_delta']:+.4f}"
            )

        # Checkpoint
        if done_count % checkpoint_every == 0:
            _write_output(
                output_path=output_path,
                metadata_base=metadata_base,
                baseline_entry=baseline_entry,
                all_results=all_results,
                num_layers=num_layers,
                done_count=done_count,
                total_configs=total_configs,
                sweep_start_time=sweep_start_time,
                final=False,
            )
            logger.debug(f"Checkpoint saved at {done_count} configs.")

    pbar.close()

    # -----------------------------------------------------------------
    # 9. Final output
    # -----------------------------------------------------------------
    _write_output(
        output_path=output_path,
        metadata_base=metadata_base,
        baseline_entry=baseline_entry,
        all_results=all_results,
        num_layers=num_layers,
        done_count=done_count,
        total_configs=total_configs,
        sweep_start_time=sweep_start_time,
        final=True,
    )
    logger.info(f"Scan complete. Results written to {output_path}")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_output(
    output_path: str,
    metadata_base: dict[str, Any],
    baseline_entry: dict[str, Any],
    all_results: list[dict[str, Any]],
    num_layers: int,
    done_count: int,
    total_configs: int,
    sweep_start_time: float,
    final: bool,
) -> None:
    import time

    elapsed = time.monotonic() - sweep_start_time
    end_utc = utc_now_iso() if final else None

    metadata = {
        **metadata_base,
        "completed_configs": done_count,
        "scan_end_utc": end_utc,
        "scan_duration_seconds": round(elapsed, 1) if final else None,
    }

    rankings = compute_rankings(all_results) if final else None
    heatmaps = build_heatmap_matrices(all_results, num_layers) if final else None

    output: dict[str, Any] = {
        "llmri_version": "1.0.0",
        "scan_metadata": metadata,
        "baseline": baseline_entry,
        "results": all_results,
    }
    if rankings:
        output["rankings"] = rankings
    if heatmaps:
        output["heatmap_matrices"] = heatmaps

    save_checkpoint(output_path, output)
