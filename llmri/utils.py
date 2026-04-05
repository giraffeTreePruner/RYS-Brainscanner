"""Utilities: logging setup, progress display, and checkpoint I/O."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> None:
    """Configure root logger.  verbose=True shows DEBUG messages."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.root.handlers = []
    logging.root.addHandler(handler)
    logging.root.setLevel(level)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_checkpoint(output_path: str | Path, data: dict[str, Any]) -> None:
    """Atomically write a JSON checkpoint to disk.

    Writes to a temp file first, then renames — avoids corrupting the output
    if the process is killed mid-write.
    """
    path = Path(output_path)
    tmp_path = path.with_suffix(".tmp.json")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    tmp_path.rename(path)


def load_checkpoint(output_path: str | Path) -> dict[str, Any] | None:
    """Load an existing checkpoint file, or return None if it doesn't exist."""
    path = Path(output_path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_completed_configs(checkpoint: dict[str, Any]) -> set[tuple[int, int]]:
    """Return the set of (i,j) configs already present in a checkpoint."""
    completed: set[tuple[int, int]] = set()
    for r in checkpoint.get("results", []):
        cfg = r.get("config")
        if cfg and len(cfg) == 2:
            completed.add((cfg[0], cfg[1]))
    return completed


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------

def make_progress_bar(total: int, desc: str = "Scanning") -> tqdm:
    return tqdm(
        total=total,
        desc=desc,
        unit="cfg",
        dynamic_ncols=True,
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Model parameter count
# ---------------------------------------------------------------------------

def count_params(model: Any) -> str:
    """Return a human-readable parameter count string like '3.09B'."""
    total = sum(p.numel() for p in model.parameters())
    if total >= 1e9:
        return f"{total / 1e9:.2f}B"
    if total >= 1e6:
        return f"{total / 1e6:.1f}M"
    return str(total)


# ---------------------------------------------------------------------------
# Post-processing: rankings and heatmap matrices
# ---------------------------------------------------------------------------

def compute_rankings(results: list[dict[str, Any]], top_n: int = 10) -> dict[str, list[list[int]]]:
    """Return top-N configs by combined, pubmedqa, and eq delta."""
    # Exclude baseline (config [0,0])
    non_baseline = [r for r in results if r["config"] != [0, 0]]

    def top(key: str) -> list[list[int]]:
        sorted_r = sorted(non_baseline, key=lambda r: r.get(key, 0.0), reverse=True)
        return [r["config"] for r in sorted_r[:top_n]]

    return {
        "top_combined": top("combined_delta"),
        "top_pubmedqa": top("pubmedqa_delta"),
        "top_eq": top("eq_delta"),
    }


def build_heatmap_matrices(
    results: list[dict[str, Any]],
    num_layers: int,
) -> dict[str, Any]:
    """Build the 2D heatmap matrices for the three delta metrics.

    matrix[i][j] = delta for config (i,j), or None if not measured.
    The matrix is (num_layers+1) x (num_layers+1) to accommodate j up to N.
    """
    size = num_layers + 1
    # Initialise with None
    pmqa = [[None] * size for _ in range(size)]
    eq = [[None] * size for _ in range(size)]
    combined = [[None] * size for _ in range(size)]

    for r in results:
        ci, cj = r["config"]
        if 0 <= ci < size and 0 <= cj < size:
            pmqa[ci][cj] = r.get("pubmedqa_delta")
            eq[ci][cj] = r.get("eq_delta")
            combined[ci][cj] = r.get("combined_delta")

    return {
        "pubmedqa_delta": {
            "description": (
                "2D array where matrix[i][j] = pubmedqa_delta for config (i,j). "
                "null if config not measured."
            ),
            "data": pmqa,
        },
        "eq_delta": {
            "description": "Same structure as pubmedqa_delta but for EQ scores.",
            "data": eq,
        },
        "combined_delta": {
            "description": "Same structure but for combined (pubmedqa + EQ) delta.",
            "data": combined,
        },
    }
