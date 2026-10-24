"""Descriptive summaries only; paired statistical tests belong to the experiment phase."""

from collections import defaultdict
from collections.abc import Iterable

from .scoring import ScoredPrediction


def accuracy_by_language(rows: Iterable[ScoredPrediction]) -> dict[str, dict[str, int | float]]:
    groups: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        groups[row.language].append(row.correct)
    return {
        language: {
            "count": len(values),
            "correct": sum(values),
            "accuracy": sum(values) / len(values),
        }
        for language, values in sorted(groups.items())
    }
