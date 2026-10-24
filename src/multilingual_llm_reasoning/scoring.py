"""Conservative exact-match scorer; no numeric or symbolic equivalence claims."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoredPrediction:
    problem_id: str
    language: str
    prediction: str
    correct: bool


def exact_match(prediction: str, reference: str) -> bool:
    return prediction.strip() == reference.strip()


# The original exact_match function remains for the synthetic plumbing test.
PARSER_VERSION = "final-answer-v2"


@dataclass(frozen=True)
class ParsedAnswer:
    answer: str | None
    status: str


def parse_final_answer(prediction: str | None) -> ParsedAnswer:
    """Parse only the strict numeric suffix after one unique FINAL_ANSWER marker.

    Version 2 permits explanatory text before the marker and one sentence-ending
    period after an integer. It never searches the explanation for a candidate answer.
    """
    from .numeric import normalize_numeric

    if not isinstance(prediction, str) or not prediction.strip():
        return ParsedAnswer(None, "empty_output")
    marker = "FINAL_ANSWER:"
    count = prediction.count(marker)
    if count == 0:
        return ParsedAnswer(None, "missing_marker")
    if count != 1:
        return ParsedAnswer(None, "multiple_markers")
    _, suffix = prediction.split(marker)
    suffix = suffix.strip()
    if suffix.endswith("."):
        candidate = suffix[:-1].strip()
        # A period is punctuation only for integer forms; preserve decimal points.
        import re

        integer = re.fullmatch(r"[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)", candidate)
        if integer:
            suffix = candidate
    try:
        return ParsedAnswer(normalize_numeric(suffix), "ok")
    except ValueError:
        return ParsedAnswer(None, "invalid_number")


def numerical_equivalence(prediction: str, reference: str) -> bool:
    """Exact decimal equality without float rounding or tolerance; invalid forms raise."""
    from .numeric import normalize_numeric

    return normalize_numeric(prediction) == normalize_numeric(reference)
