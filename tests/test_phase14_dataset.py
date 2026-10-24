"""Phase 14 selection and pending-review artifact tests use local fixtures only."""

import csv
import json
from pathlib import Path

import pytest

from multilingual_llm_reasoning.phase14_dataset import (
    SOURCE_EXCLUSIONS,
    TADPOLES_ID,
    _jsonl,
    build_variant_records,
    export_phase14_draft,
    select_questions,
    validate_variant_records,
)
from multilingual_llm_reasoning.preparation import original_problem_id
from multilingual_llm_reasoning.translation import LANGUAGES
from multilingual_llm_reasoning.translation_qa import (
    duplicate_translation_flags,
    qa_flags,
)


def rows(size: int = 130) -> list[dict[str, object]]:
    records = []
    for index in range(size):
        question = f"A person has {index + 1} apples and buys 2 more. How many apples?"
        records.append(
            {
                "original_problem_id": original_problem_id(question),
                "source_split": "test",
                "split": "test",
                "original_question": question,
                "gold_answer": str(index + 3),
            }
        )
    # Keep the known excluded source in the test universe so it cannot leak into selection.
    tadpole_question = "Finn watches 11 tadpoles swimming in the pond."
    records.append(
        {
            "original_problem_id": TADPOLES_ID,
            "source_split": "test",
            "split": "test",
            "original_question": tadpole_question,
            "gold_answer": "15",
        }
    )
    return records


def test_seeded_selection_is_repeatable_and_filters_curated_ids() -> None:
    source = rows()
    first = select_questions(source, seed=42, size=100)
    second = select_questions(source, seed=42, size=100)
    assert [row["original_problem_id"] for row in first] == [
        row["original_problem_id"] for row in second
    ]
    assert len(first) == 100
    assert all(row["original_problem_id"] not in SOURCE_EXCLUSIONS for row in first)


def test_four_language_pending_dataset_and_split_integrity() -> None:
    questions = select_questions(rows(), seed=42, size=10)
    variants = build_variant_records(questions)
    report = validate_variant_records(
        questions, variants, training_ids={"train-id"}, validation_ids={"validation-id"}
    )
    assert len(variants) == 40
    assert report["variants_by_language"] == {language: 10 for language in LANGUAGES}
    assert report["training_overlap"] == report["validation_overlap"] == 0
    assert report["missing_non_english_translations"] == 30
    assert report["approved_non_english_translations"] == 0
    assert report["research_ready"] is False
    assert all(
        row["approval_status"] == "pending" and row["translated_text"] == ""
        for row in variants
        if row["language"] != "en"
    )


def test_split_leakage_and_approval_are_rejected() -> None:
    questions = select_questions(rows(), size=2)
    variants = build_variant_records(questions)
    with pytest.raises(ValueError, match="overlap"):
        validate_variant_records(
            questions,
            variants,
            training_ids={questions[0]["original_problem_id"]},
            validation_ids=set(),
        )


def test_research_readiness_requires_nonempty_approved_translations() -> None:
    questions = select_questions(rows(), size=1)
    variants = build_variant_records(questions)
    pending = validate_variant_records(
        questions, variants, training_ids=set(), validation_ids=set()
    )
    assert pending["research_ready"] is False
    for row in variants:
        if row["language"] != "en":
            row["translated_text"] = f"Reviewed text {row['language']}"
            row["approval_status"] = "approved"
    approved = validate_variant_records(
        questions, variants, training_ids=set(), validation_ids=set()
    )
    assert approved["research_ready"] is True


def test_export_is_hash_audited_and_review_csv_keeps_all_rows_pending(tmp_path: Path) -> None:
    questions = select_questions(rows(), size=2)
    variants = build_variant_records(questions)
    manifest_path = export_phase14_draft(
        tmp_path / "phase14",
        questions=questions,
        variants=variants,
        source_manifest={
            "dataset": {
                "resolved_revision": "a" * 40,
                "source_sha256": {"test": "c" * 64},
            },
            "splits": {"test": {"sha256": "d" * 64}},
        },
        source_manifest_sha256="b" * 64,
        train_ids=set(),
        validation_ids=set(),
        translation_blocker="local engine unavailable",
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["research_ready"] is False
    assert manifest["translation"]["paid_api_requests_made"] == 0
    review_path = manifest_path.parent / "human_review.csv"
    with review_path.open(encoding="utf-8", newline="") as stream:
        review = list(csv.DictReader(stream))
    assert len(review) == 6
    assert {row["approval_status"] for row in review} == {"pending"}
    for artifact in manifest["artifacts"].values():
        content = (manifest_path.parent / artifact["path"]).read_bytes()
        import hashlib

        assert hashlib.sha256(content).hexdigest() == artifact["sha256"]


def test_translation_heuristics_report_numeric_units_relations_and_leakage() -> None:
    changed = qa_flags("She has 5 kg of flour.", "Sie hat 6 kg Mehl.", "de")
    assert "changed_numeric_tokens" in changed
    relation = qa_flags("He has twice as many apples.", "彼はりんごを2倍持っています。", "ja")
    assert "relational_cues_not_detected" not in relation
    units = qa_flags("It costs $5.", "Es kostet 5 Euro.", "de")
    assert "unit_currency_token_mismatch" in units
    leakage = qa_flags("How many?", "答案是 FINAL_ANSWER: 5", "zh")
    assert "possible_solution_leakage" in leakage
    assert "missing_target_script" not in leakage
    leaked_answer = qa_flags("How many?", "Die Antwort ist 5.", "de", gold_answer="5")
    assert "possible_answer_leakage" in leaked_answer
    source_contains_answer = qa_flags(
        "There are 5 apples.", "Es gibt 5 Äpfel.", "de", gold_answer="5"
    )
    assert "possible_answer_leakage" not in source_contains_answer


def test_duplicate_translation_detection_is_a_flag_not_an_approval() -> None:
    variants = [
        {"original_problem_id": "id-1", "language": "de", "translated_text": "Frage 1"},
        {"original_problem_id": "id-2", "language": "de", "translated_text": " Frage 1 "},
    ]
    assert duplicate_translation_flags(variants) == {
        ("id-1", "de"): "duplicate_translated_question",
        ("id-2", "de"): "duplicate_translated_question",
    }


def test_phase14_jsonl_escapes_unicode_line_separators() -> None:
    content = _jsonl([{"question": "before\u2028after\u2029end"}])
    assert b"\\u2028" in content and b"\\u2029" in content
    assert len(content.decode().splitlines()) == 1
    assert json.loads(content.splitlines()[0]) == {"question": "before\u2028after\u2029end"}
