"""BrainScan CLI entry point.

Commands:
  brainscan scan      — run the (i,j) sweep on a model
  brainscan convert   — convert RYS pickle files to BrainScan JSON
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

logger = logging.getLogger(__name__)

# Default dataset paths (bundled with the package)
_PKG_ROOT = Path(__file__).parent
_DATASETS_DIR = _PKG_ROOT.parent / "datasets"
_DEFAULT_PUBMEDQA = _DATASETS_DIR / "pubmedqa_16.json"
_DEFAULT_EQ = _DATASETS_DIR / "eq_16.json"


@click.group()
def cli() -> None:
    """BrainScan — RYS layer-duplication sweep tool."""


# ---------------------------------------------------------------------------
# brainscan scan
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--model", "-m",
    required=True,
    help="HuggingFace model ID or path to a local checkpoint.",
)
@click.option(
    "--output", "-o",
    default="scan_results.json",
    show_default=True,
    help="Path to write the scan results JSON.",
)
@click.option(
    "--backend",
    type=click.Choice(["hf", "exllama"], case_sensitive=False),
    default="hf",
    show_default=True,
    help=(
        '"hf" uses HuggingFace transformers (works on Mac/MPS, CPU, CUDA). '
        '"exllama" requires an NVIDIA GPU and ExLlamaV2 installed manually.'
    ),
)
@click.option(
    "--device",
    default="auto",
    show_default=True,
    help='Compute device: "cuda", "mps", "cpu", or "auto" (auto-detect).',
)
@click.option(
    "--probes",
    default="pubmedqa,eq",
    show_default=True,
    help='Comma-separated probe sets to run: "pubmedqa", "eq", or "pubmedqa,eq".',
)
@click.option(
    "--pubmedqa-dataset",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help=(
        "Path to a custom PubMedQA probe JSON file. "
        f"Defaults to the bundled {_DEFAULT_PUBMEDQA.name}."
    ),
)
@click.option(
    "--eq-dataset",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help=(
        "Path to a custom EQ-Bench probe JSON file. "
        f"Defaults to the bundled {_DEFAULT_EQ.name}."
    ),
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help=(
        "Resume an interrupted scan. Reads --output and skips configs already "
        "recorded there (matched by (i,j) pair)."
    ),
)
@click.option(
    "--max-new-tokens",
    default=16,
    show_default=True,
    help=(
        "Maximum tokens to generate per probe. PubMedQA needs 1-3; "
        "EQ-Bench needs more — the backend automatically uses max(this, 128) "
        "for EQ probes."
    ),
)
@click.option(
    "--checkpoint-every",
    default=20,
    show_default=True,
    help="Save a checkpoint to --output every N completed configs.",
)
@click.option(
    "--cache-dir",
    default=None,
    type=click.Path(file_okay=False),
    envvar="BRAINSCAN_CACHE_DIR",
    help=(
        "Directory to cache downloaded model files. "
        "Defaults to the standard HuggingFace hub cache (~/.cache/huggingface/hub/). "
        "Can also be set via the BRAINSCAN_CACHE_DIR environment variable."
    ),
)
@click.option(
    "--offline",
    is_flag=True,
    default=False,
    envvar="BRAINSCAN_OFFLINE",
    help=(
        "Run in offline mode: never hit the network, use only locally cached model files. "
        "Raises an error if the model has not been downloaded yet. "
        "Useful for repeated sweeps after the first download."
    ),
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Print per-config scores as they complete.",
)
def scan(
    model: str,
    output: str,
    backend: str,
    device: str,
    probes: str,
    pubmedqa_dataset: str | None,
    eq_dataset: str | None,
    resume: bool,
    max_new_tokens: int,
    checkpoint_every: int,
    cache_dir: str | None,
    offline: bool,
    verbose: bool,
) -> None:
    """Run the RYS (i,j) layer-duplication sweep on MODEL.

    Produces a single JSON file at --output that the BrainScan Viewer can
    consume to render interactive heatmaps.

    Examples:

    \b
    # Minimal — scan a HuggingFace model (downloads to HF cache on first run)
    brainscan scan --model Qwen/Qwen2.5-3B-Instruct

    \b
    # Resume an interrupted scan without re-downloading
    brainscan scan --model Qwen/Qwen2.5-3B-Instruct --output scan.json --resume --offline

    \b
    # Use a custom cache directory
    brainscan scan --model Qwen/Qwen2.5-3B-Instruct --cache-dir ~/models

    \b
    # PubMedQA probes only, force MPS device
    brainscan scan --model /path/to/model --probes pubmedqa --device mps
    """
    from brainscan.utils import setup_logging
    setup_logging(verbose=verbose)

    # Parse probe set
    active_probes: set[str] = set()
    for p in probes.split(","):
        p = p.strip().lower()
        if p not in ("pubmedqa", "eq"):
            raise click.BadParameter(
                f"Unknown probe set {p!r}. Choose from: pubmedqa, eq",
                param_hint="--probes",
            )
        active_probes.add(p)

    # Resolve dataset paths
    pubmedqa_path = pubmedqa_dataset or str(_DEFAULT_PUBMEDQA)
    eq_path = eq_dataset or str(_DEFAULT_EQ)

    if "pubmedqa" in active_probes and not Path(pubmedqa_path).exists():
        raise click.UsageError(
            f"PubMedQA dataset not found at {pubmedqa_path!r}.\n"
            "Run: brainscan create-dataset --pubmedqa\n"
            "Or supply a custom path with --pubmedqa-dataset."
        )
    if "eq" in active_probes and not Path(eq_path).exists():
        raise click.UsageError(
            f"EQ dataset not found at {eq_path!r}.\n"
            "Make sure datasets/eq_16.json is present in the package."
        )

    from brainscan.scanner import run_scan

    try:
        run_scan(
            model_path=model,
            output_path=output,
            backend=backend.lower(),
            device=device.lower(),
            probes=active_probes,
            pubmedqa_dataset_path=pubmedqa_path,
            eq_dataset_path=eq_path,
            resume=resume,
            checkpoint_every=checkpoint_every,
            max_new_tokens=max_new_tokens,
            cache_dir=cache_dir,
            offline=offline,
            verbose=verbose,
        )
    except KeyboardInterrupt:
        click.echo("\nScan interrupted. Results up to this point are saved.", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# brainscan convert
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--pkl-pubmedqa",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a RYS pubmedqa .pkl results file (or a math .pkl — see note).",
)
@click.option(
    "--pkl-eq",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a RYS eq .pkl results file.",
)
@click.option(
    "--model-name",
    required=True,
    help='Model name string to embed in the output, e.g. "Qwen/Qwen2.5-3B-Instruct".',
)
@click.option(
    "--num-layers",
    required=True,
    type=int,
    help="Number of transformer layers in the model.",
)
@click.option(
    "--output", "-o",
    default="scan_results.json",
    show_default=True,
    help="Path to write the converted BrainScan JSON.",
)
def convert(
    pkl_pubmedqa: str | None,
    pkl_eq: str | None,
    model_name: str,
    num_layers: int,
    output: str,
) -> None:
    """Convert RYS repo pickle files to BrainScan JSON format.

    The RYS repo outputs separate .pkl files for each probe type.  Use this
    command to combine them into a single scan_results.json for the viewer.

    Note: RYS math .pkl files store "math_score" rather than "pubmedqa_score".
    When converting, the math score is mapped to the pubmedqa_score field and
    a "legacy_math_score" flag is added to scan_metadata so the viewer knows
    the axis label should say "Math" rather than "PubMedQA".
    """
    import pickle
    import json
    from brainscan.relayer import build_layer_path, get_duplicated_layers
    from brainscan.utils import build_heatmap_matrices, compute_rankings, save_checkpoint, utc_now_iso

    if not pkl_pubmedqa and not pkl_eq:
        raise click.UsageError("Provide at least one of --pkl-pubmedqa or --pkl-eq.")

    # Load pickle files
    pubmedqa_data: dict = {}
    eq_data: dict = {}
    is_legacy_math = False

    if pkl_pubmedqa:
        with open(pkl_pubmedqa, "rb") as f:
            raw = pickle.load(f)
        # RYS pickles are dicts keyed by (i,j) tuple
        # Values may be floats or dicts — handle both
        for key, val in raw.items():
            score = val if isinstance(val, float) else val.get("score", val.get("math_score", 0.0))
            pubmedqa_data[key] = score
        # Detect if this is a math pkl (heuristic: check keys in raw)
        if any(
            "math" in str(k).lower() or "math_score" in (str(val) if not isinstance(val, float) else "")
            for k, val in raw.items()
        ):
            is_legacy_math = True

    if pkl_eq:
        with open(pkl_eq, "rb") as f:
            raw = pickle.load(f)
        for key, val in raw.items():
            score = val if isinstance(val, float) else val.get("score", val.get("eq_score", 0.0))
            eq_data[key] = score

    # Merge into results list
    all_keys: set = set(pubmedqa_data.keys()) | set(eq_data.keys())
    baseline_pubmedqa = pubmedqa_data.get((0, 0), 0.0)
    baseline_eq = eq_data.get((0, 0), 0.0)
    baseline_combined = (baseline_pubmedqa + baseline_eq) / 2.0

    results = []
    for key in sorted(all_keys, key=lambda k: (k[0], k[1])):
        i, j = key
        pmqa = pubmedqa_data.get(key, 0.0)
        eq = eq_data.get(key, 0.0)
        combined = (pmqa + eq) / 2.0
        layer_path = build_layer_path(i, j, num_layers)
        dup = get_duplicated_layers(i, j)
        results.append({
            "config": [i, j],
            "pubmedqa_score": pmqa,
            "eq_score": eq,
            "combined_score": combined,
            "pubmedqa_delta": round(pmqa - baseline_pubmedqa, 6),
            "eq_delta": round(eq - baseline_eq, 6),
            "combined_delta": round(combined - baseline_combined, 6),
            "duplicated_layers": dup,
            "num_duplicated": len(dup),
            "layer_path": layer_path,
            "total_layers_in_path": len(layer_path),
            "param_increase_pct": round(len(dup) / num_layers * 100, 2),
        })

    baseline_entry = {
        "config": [0, 0],
        "pubmedqa_score": baseline_pubmedqa,
        "eq_score": baseline_eq,
        "combined_score": baseline_combined,
    }

    metadata = {
        "model_name": model_name,
        "model_type": "unknown",
        "num_layers": num_layers,
        "hidden_size": 0,
        "num_attention_heads": 0,
        "num_key_value_heads": None,
        "total_params_base": "unknown",
        "backend": "rys-pkl-import",
        "device": "unknown",
        "scan_start_utc": utc_now_iso(),
        "scan_end_utc": utc_now_iso(),
        "scan_duration_seconds": None,
        "total_configs": len(results),
        "completed_configs": len(results),
        "pubmedqa_dataset": str(pkl_pubmedqa or ""),
        "pubmedqa_dataset_size": 0,
        "eq_dataset": str(pkl_eq or ""),
        "eq_dataset_size": 0,
        "max_new_tokens": 0,
        "legacy_math_score": is_legacy_math,
    }

    out = {
        "brainscan_version": "1.0.0",
        "scan_metadata": metadata,
        "baseline": baseline_entry,
        "results": results,
        "rankings": compute_rankings(results),
        "heatmap_matrices": build_heatmap_matrices(results, num_layers),
    }

    save_checkpoint(output, out)
    click.echo(f"Converted {len(results)} configs → {output}")


# ---------------------------------------------------------------------------
# brainscan create-dataset
# ---------------------------------------------------------------------------

@cli.command("create-dataset")
@click.option(
    "--pubmedqa",
    "create_pubmedqa",
    is_flag=True,
    default=False,
    help="Download and create datasets/pubmedqa_16.json and pubmedqa_100.json.",
)
@click.option(
    "--output-dir",
    default=str(_DATASETS_DIR),
    show_default=True,
    help="Directory to write the dataset files.",
)
@click.option(
    "--seed",
    default=42,
    show_default=True,
    help="Random seed for question selection.",
)
def create_dataset(create_pubmedqa: bool, output_dir: str, seed: int) -> None:
    """Download and create the bundled probe dataset files.

    Requires the 'datasets' extra: pip install brainscan[datasets]

    This only needs to be run once after installation.  The eq_16.json file
    is already bundled; only pubmedqa_16.json needs to be downloaded.
    """
    if not create_pubmedqa:
        click.echo("Nothing to do. Use --pubmedqa to create the PubMedQA datasets.")
        return

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise click.UsageError(
            "The 'datasets' package is required. Install it with:\n"
            "  pip install brainscan[datasets]\n"
            "  or: pip install datasets"
        )

    import random
    import json

    click.echo("Downloading qiaojin/PubMedQA (pqa_labeled split) ...")
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

    yes_qs = [q for q in ds if q["final_decision"] == "yes"]
    no_qs = [q for q in ds if q["final_decision"] == "no"]
    maybe_qs = [q for q in ds if q["final_decision"] == "maybe"]

    random.seed(seed)

    # 16-question sweep set: 7 yes, 5 no, 4 maybe
    probes_16 = (
        random.sample(yes_qs, 7)
        + random.sample(no_qs, 5)
        + random.sample(maybe_qs, 4)
    )
    random.shuffle(probes_16)

    # 100-question validation set: 45 yes, 30 no, 25 maybe
    probes_100 = (
        random.sample(yes_qs, 45)
        + random.sample(no_qs, 30)
        + random.sample(maybe_qs, 25)
    )
    random.shuffle(probes_100)

    def fmt(probes: list) -> list[dict]:
        out = []
        for q in probes:
            context = " ".join(q["context"]["contexts"])
            out.append({
                "id": str(q["pubid"]),
                "prompt": (
                    f"Context: {context}\n\n"
                    f"Question: {q['question']}\n\n"
                    f"Answer with just yes, no, or maybe:"
                ),
                "answer": q["final_decision"],
                "type": "pubmedqa",
            })
        return out

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p16 = out_dir / "pubmedqa_16.json"
    p100 = out_dir / "pubmedqa_100.json"

    json.dump(fmt(probes_16), open(p16, "w"), indent=2)
    click.echo(f"Created {p16} ({len(probes_16)} questions)")

    json.dump(fmt(probes_100), open(p100, "w"), indent=2)
    click.echo(f"Created {p100} ({len(probes_100)} questions)")


if __name__ == "__main__":
    cli()
