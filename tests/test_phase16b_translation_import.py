"""Structural guards for the fixed training-translation import."""

import csv
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "phase16b_training_translations",
    Path(__file__).resolve().parents[1] / "scripts/phase16b_training_translations.py",
)
assert spec and spec.loader
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


def _row() -> dict[str, str]:
    return {
        "canonical_problem_id": "training-id-1",
        "target_language": "de",
        "english_original": "Ana has 2 apples and 3 oranges. How many fruits?",
        "english_worked_solution": "2 + 3 = 5\n#### 5",
        "gold_answer": "5",
        "source_sha256": handoff.digest("Ana has 2 apples and 3 oranges. How many fruits?"),
        "translated_question": "Ana hat 2 Äpfel und 3 Orangen. Wie viele Früchte?",
        "numeric_qa_note": "",
        "mathematical_equivalence": "pending",
        "reviewer_comments": "",
        "approval_status": "pending",
    }


def _write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=handoff.FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_valid_pending_translation_is_not_training_ready(tmp_path, monkeypatch):
    row = _row()
    monkeypatch.setattr(handoff, "source_rows", lambda: [row])
    path = tmp_path / "review.csv"
    _write(path, [row])
    result = handoff.validate(path)
    assert result["issues"] == []
    assert result["ready"] is False


def test_import_rejects_duplicates_leakage_and_unexplained_numeric_change(tmp_path, monkeypatch):
    row = _row()
    monkeypatch.setattr(handoff, "source_rows", lambda: [row])
    changed = dict(row, translated_question="Ana hat 2 Äpfel. Wie viele Früchte?")
    leaked = dict(row, canonical_problem_id="evaluation-id")
    path = tmp_path / "review.csv"
    _write(path, [changed, changed, leaked])
    issues = {item["issue"] for item in handoff.validate(path)["issues"]}
    assert {
        "duplicate_id_language",
        "unexpected_or_leaked_id",
        "numeric_tokens_differ_review_required",
    } <= issues
