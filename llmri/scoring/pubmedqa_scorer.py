"""PubMedQA yes/no/maybe scorer.

Each probe asks the model to classify a biomedical abstract as yes/no/maybe.
Scoring is exact-match: 1.0 if the first valid keyword in the response matches
the ground-truth label, 0.0 otherwise.

The output is intentionally only 1-3 tokens, which is why PubMedQA makes an
ideal fast probe for the (i,j) sweep.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


VALID_ANSWERS = ("yes", "no", "maybe")


def score_pubmedqa(response: str, correct_answer: str) -> float:
    """Return 1.0 if the model's response matches the correct answer, else 0.0.

    Checks the first 20 characters of the stripped response for any of the
    three valid answer words, case-insensitively.  Models may respond with
    "Yes.", "Yes, because...", or just "yes" — all are handled.
    """
    snippet = response.strip().lower()[:20]
    for answer in VALID_ANSWERS:
        if answer in snippet:
            return 1.0 if answer == correct_answer.lower() else 0.0
    return 0.0  # no valid answer found in the snippet


def score_pubmedqa_batch(
    responses: list[str],
    probes: list[dict[str, Any]],
) -> float:
    """Return the mean accuracy across a batch of PubMedQA responses.

    Args:
        responses: Raw text outputs from the model, one per probe.
        probes: List of probe dicts with an "answer" key (the ground truth).

    Returns:
        Mean score in [0.0, 1.0].
    """
    if not probes:
        return 0.0
    scores = [
        score_pubmedqa(resp, probe["answer"])
        for resp, probe in zip(responses, probes)
    ]
    return sum(scores) / len(scores)


def load_pubmedqa_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load a PubMedQA probe file and return the list of probe dicts.

    Expected format per entry:
        {
            "id": "12345",
            "prompt": "Context: ...\n\nQuestion: ...\n\nAnswer with just yes, no, or maybe:",
            "answer": "yes",
            "type": "pubmedqa"
        }
    """
    with open(path) as f:
        probes = json.load(f)

    for p in probes:
        if "prompt" not in p or "answer" not in p:
            raise ValueError(
                f"Probe {p.get('id', '?')} is missing 'prompt' or 'answer' keys."
            )
        if p["answer"].lower() not in VALID_ANSWERS:
            raise ValueError(
                f"Probe {p.get('id', '?')} has invalid answer: {p['answer']!r}. "
                f"Expected one of {VALID_ANSWERS}."
            )

    return probes
