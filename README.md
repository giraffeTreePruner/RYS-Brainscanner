# BrainScan

BrainScan runs exhaustive layer-duplication sweeps on transformer models to measure how duplicating different layer ranges affects model performance. For every valid `(i, j)` pair in an N-layer model, it constructs a temporary layer path where layers `i` through `j−1` execute twice, scores the resulting model on two probe sets, and records the results in a structured JSON file.

---

## What it does

For a model with N layers, BrainScan evaluates N×(N+1)/2 + 1 configurations (including a no-duplication baseline). Each configuration is scored on:

- **PubMedQA** — biomedical yes/no/maybe question answering (accuracy)
- **EQ-Bench** — emotional intelligence dialogues (MAE-based score with confidence weighting)

Outputs include per-configuration scores, deltas from baseline, top-10 rankings by each metric, and 2D heatmap matrices ready for visualization.

---

## Features

- Full `(i, j)` sweep with no weight copies — uses shallow layer references to avoid memory duplication
- Resume interrupted scans with `--resume`; atomic checkpointing every N configs
- Pluggable inference backends (HuggingFace Transformers primary; ExLlama stub for future CUDA use)
- Auto-detects layer paths for Llama, Qwen, Mistral, GPT-2, Falcon, MPT, and variants
- Bundled 16-question probe sets for fast sweeps; 100-question PubMedQA set for post-sweep validation
- Legacy RYS pickle import via `brainscan convert`
- Output includes rankings and heatmap matrices for immediate analysis

---

## Installation

Requires Python 3.10+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone <repo-url>
cd brainscan
uv sync
```

For GPU inference, ensure a CUDA-enabled PyTorch build is installed before running `uv sync`.

To generate the bundled PubMedQA dataset from the HuggingFace hub (optional — already included):

```bash
uv run brainscan create-dataset
```

---

## Usage

### Run a sweep

```bash
uv run brainscan scan \
  --model Qwen/Qwen2.5-3B-Instruct \
  --output model_scans/my-scan.json
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | _(required)_ | HuggingFace model ID or local path |
| `--output` | `scan_results.json` | Output JSON file |
| `--backend` | `hf` | Inference backend (`hf`) |
| `--device` | auto-detected | `cuda`, `mps`, or `cpu` |
| `--probes` | `pubmedqa,eq` | Comma-separated probe sets to run |
| `--pubmedqa-dataset` | bundled | Path to PubMedQA dataset JSON |
| `--eq-dataset` | bundled | Path to EQ-Bench dataset JSON |
| `--resume` | off | Resume from an existing output file |
| `--checkpoint-every` | `20` | Save progress every N configs |

### Resume an interrupted scan

```bash
uv run brainscan scan --model Qwen/Qwen2.5-3B-Instruct --output scan.json --resume
```

### Convert a legacy RYS pickle file

```bash
uv run brainscan convert rys_results.pkl --output scan.json --probe pubmedqa
```

---

## Directory structure

```
brainscan/
├── brainscan/                  # Main package
│   ├── cli.py                  # Click CLI entry points
│   ├── scanner.py              # Sweep orchestrator
│   ├── relayer.py              # Layer path construction and model patching
│   ├── schema.py               # Pydantic output schema
│   ├── utils.py                # Logging, checkpointing, ranking, heatmap utilities
│   ├── backends/
│   │   ├── hf_backend.py       # HuggingFace Transformers backend
│   │   └── exllama_backend.py  # ExLlama backend (stub)
│   └── scoring/
│       ├── pubmedqa_scorer.py  # Accuracy scorer for PubMedQA
│       └── eq_scorer.py        # MAE-based scorer for EQ-Bench
├── datasets/
│   ├── manifest.json           # Dataset metadata
│   ├── eq_16.json              # 16 EQ-Bench scenarios (bundled)
│   ├── pubmedqa_16.json        # 16 PubMedQA questions (sweep set)
│   └── pubmedqa_100.json       # 100 PubMedQA questions (validation)
├── model_scans/                # Pre-computed scan outputs
│   ├── Qwen2-5-Instruct-3B.json
│   └── llama3-2-Instruct-3B.json
└── pyproject.toml
```

---

## Output format

Each scan produces a single JSON file conforming to the following structure:

```jsonc
{
  "brainscan_version": "1.0.0",
  "scan_metadata": {
    "model_name": "Qwen/Qwen2.5-3B-Instruct",
    "num_layers": 36,
    "total_configs": 667,
    "completed_configs": 667,
    "scan_duration_seconds": 5280.0,
    // ...
  },
  "baseline": {
    "config": [0, 0],
    "pubmedqa_score": 0.25,
    "eq_score": 0.524,
    "combined_score": 0.387
  },
  "results": [
    {
      "config": [0, 1],          // (i, j): duplicate layers i..j-1
      "pubmedqa_score": 0.5,
      "eq_score": 0.524,
      "combined_score": 0.512,
      "pubmedqa_delta": 0.25,    // score - baseline
      "eq_delta": 0.0,
      "combined_delta": 0.125,
      "duplicated_layers": [0],
      "num_duplicated": 1,
      "layer_path": [0, 0, 1, 2, "..."],
      "total_layers_in_path": 37,
      "param_increase_pct": 2.78
    }
    // N*(N+1)/2 more entries
  ],
  "rankings": {
    "top_combined": [[3, 7], [2, 6], "..."],  // top-10 configs
    "top_pubmedqa": ["..."],
    "top_eq": ["..."]
  },
  "heatmap_matrices": {
    "combined_delta": {
      "description": "matrix[i][j] = combined_delta for config (i,j)",
      "data": [[null, 0.125, "..."], "..."]   // null where j <= i
    },
    "pubmedqa_delta": { "..." },
    "eq_delta": { "..." }
  }
}
```

**Score definitions:**

| Field | Range | Description |
|-------|-------|-------------|
| `pubmedqa_score` | 0–1 | Fraction of correct yes/no/maybe answers |
| `eq_score` | 0–1 | `1 - (MAE / 40)`, confidence-weighted toward 0.5 on parse failure |
| `combined_score` | 0–1 | Simple average of the two scores |
| `*_delta` | −1 to 1 | Score minus baseline |

**Layer path semantics:**

Config `(i, j)` produces path `[0 .. j-1] + [i .. N-1]`, so layers `i` through `j-1` execute twice. Config `(0, 0)` is the unmodified baseline.

---

## Tech stack

| Component | Library |
|-----------|---------|
| CLI | [Click](https://click.palletsprojects.com/) |
| Model inference | [Transformers](https://github.com/huggingface/transformers) |
| Deep learning | [PyTorch](https://pytorch.org/) |
| Schema validation | [Pydantic v2](https://docs.pydantic.dev/) |
| Numerical ops | [NumPy](https://numpy.org/) |
| Progress display | [tqdm](https://github.com/tqdm/tqdm) |
| Build system | [Hatchling](https://hatch.pypa.io/) / [uv](https://github.com/astral-sh/uv) |

---

## Pre-computed scans

Two reference scans are included in `model_scans/`:

| File | Model | Configs |
|------|-------|---------|
| `Qwen2-5-Instruct-3B.json` | Qwen/Qwen2.5-3B-Instruct | 667 |
| `llama3-2-Instruct-3B.json` | meta-llama/Llama-3.2-3B-Instruct | 667 |
