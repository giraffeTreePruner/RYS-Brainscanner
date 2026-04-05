"""EQ-Bench scorer for emotional intelligence probes.

Each probe presents a dialogue and asks the model to rate the intensity of four
named emotions (0-10 scale) for a specific character.  The output format is:

    First pass scores:
    Emotion1: <score>
    Emotion2: <score>
    Emotion3: <score>
    Emotion4: <score>

Scoring approach (matches dnhkng/RYS repo)
-------------------------------------------
Uses MAE-based scoring, NOT correlation.  Per the RYS implementation:

    total_diff = sum(|predicted_score - reference_score|) for each of the 4 emotions
    raw_score  = 1.0 - (total_diff / 40.0)
        where 40 = 4 emotions × 10-point max scale

Confidence weighting pulls uncertain predictions toward a neutral baseline:
    final_score = (confidence × raw_score) + ((1 - confidence) × 0.5)

Confidence levels:
    1.0  — both "First pass scores:" and "Revised scores:" parsed successfully
    0.9  — only "Revised scores:" found
    0.8  — only "First pass scores:" found (our typical case for short outputs)
    0.5  — some numbers found but format didn't match cleanly
    0.0  — parse failure → returns 0.5 (the neutral fallback)

When both first-pass and revised scores exist, they're averaged:
    combined = 0.5 * first_pass + 0.5 * revised

Note: the reference_answer in eq_16.json uses normalized scores (fullscale
integers divided by their sum * 10, so they sum to ~10).  We compare predicted
raw 0-10 scores directly against these normalized references — a slight
approximation that matches the spirit of the RYS scoring.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _extract_block(response: str, marker: str, stop_markers: list[str]) -> str | None:
    """Extract the text block that starts with marker and ends at the first stop_marker."""
    lower = response.lower()
    idx = lower.find(marker.lower())
    if idx == -1:
        return None
    block = response[idx:]
    end_idx = len(block)
    for stop in stop_markers:
        pos = block.lower().find(stop.lower())
        if pos != -1:
            end_idx = min(end_idx, pos)
    return block[:end_idx]


def _parse_scores_from_block(block: str, emotion_names: list[str]) -> dict[str, float] | None:
    """Extract emotion → score mappings from a scores block."""
    scores: dict[str, float] = {}
    for name in emotion_names:
        pattern = re.compile(
            rf"{re.escape(name)}\s*:\s*([0-9]+(?:\.[0-9]+)?)",
            re.IGNORECASE,
        )
        m = pattern.search(block)
        if m:
            val = float(m.group(1))
            scores[name] = min(val, 10.0)  # cap at 10
    if len(scores) < len(emotion_names):
        return None
    return scores


def parse_eq_response(
    response: str,
    emotion_names: list[str],
) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    """Parse first-pass and (optionally) revised scores from the model output.

    Returns (first_pass_scores, revised_scores).  Either may be None if not found.
    """
    stop = ["\n\n", "[end"]

    first_block = _extract_block(response, "first pass scores:", stop + ["revised scores"])
    first = _parse_scores_from_block(first_block, emotion_names) if first_block else None

    revised_block = _extract_block(response, "revised scores:", stop)
    revised = _parse_scores_from_block(revised_block, emotion_names) if revised_block else None

    return first, revised


# ---------------------------------------------------------------------------
# Scoring (MAE-based, matching dnhkng/RYS)
# ---------------------------------------------------------------------------

def _mae_score(pred: dict[str, float], ref_scores: list[float], emotion_names: list[str]) -> float:
    """Compute raw MAE-based score for one set of predicted emotion scores."""
    total_diff = sum(
        abs(pred.get(name, 0.0) - ref)
        for name, ref in zip(emotion_names, ref_scores)
    )
    return 1.0 - (total_diff / 40.0)  # 40 = 4 emotions × 10-point max


def score_eq_scenario(response: str, reference: dict[str, Any]) -> float:
    """Score a single EQ-Bench scenario using MAE-based scoring (matches RYS repo).

    Args:
        response: Raw model output text.
        reference: The reference_answer dict from eq_16.json.

    Returns:
        Confidence-weighted score in [0.0, 1.0].
        Returns 0.5 on parse failure (neutral baseline).
    """
    emotion_names = [reference[f"emotion{i}"] for i in range(1, 5)]
    ref_scores = [reference[f"emotion{i}_score"] for i in range(1, 5)]

    first, revised = parse_eq_response(response, emotion_names)

    if first is None and revised is None:
        return 0.5  # parse failure → neutral

    if first is not None and revised is not None:
        # Both found: average, full confidence
        s1 = _mae_score(first, ref_scores, emotion_names)
        s2 = _mae_score(revised, ref_scores, emotion_names)
        raw_score = 0.5 * s1 + 0.5 * s2
        confidence = 1.0
    elif revised is not None:
        raw_score = _mae_score(revised, ref_scores, emotion_names)
        confidence = 0.9
    else:
        raw_score = _mae_score(first, ref_scores, emotion_names)  # type: ignore[arg-type]
        confidence = 0.8

    return (confidence * raw_score) + ((1.0 - confidence) * 0.5)


def score_eq_batch(
    responses: list[str],
    probes: list[dict[str, Any]],
) -> float:
    """Return mean EQ score across a batch of scenarios.

    Args:
        responses: Raw model outputs, one per probe.
        probes: List of probe dicts from eq_16.json (each has "prompt" and
                "reference_answer" keys).

    Returns:
        Mean Pearson correlation in [0.0, 1.0].
    """
    if not probes:
        return 0.0
    scores = [
        score_eq_scenario(resp, probe["reference_answer"])
        for resp, probe in zip(responses, probes)
    ]
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_eq_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load an EQ-Bench probe file (eq_16.json format from dnhkng/RYS).

    The file is a dict keyed by string integers ("1", "2", ...).  Each entry
    has "prompt" and "reference_answer" sub-dicts.

    Returns a flat list of probe dicts with keys:
        "id", "prompt", "reference_answer"
    """
    with open(path) as f:
        raw = json.load(f)

    probes: list[dict[str, Any]] = []
    for key in sorted(raw.keys(), key=lambda k: int(k)):
        entry = raw[key]
        if "prompt" not in entry or "reference_answer" not in entry:
            raise ValueError(f"EQ probe {key!r} is missing 'prompt' or 'reference_answer'.")
        probes.append({
            "id": key,
            "prompt": entry["prompt"],
            "reference_answer": entry["reference_answer"],
        })

    return probes
