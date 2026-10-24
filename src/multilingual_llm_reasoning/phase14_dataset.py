"""Auditable Phase 14 selection and review-package export helpers."""

import csv
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .translation import LANGUAGES
from .translation_qa import duplicate_translation_flags

Record = dict[str, Any]
DATASET_VERSION = "phase14-gsm8k-test-100-v1"
SELECTION_PURPOSE = "phase14-test-v1"
TADPOLES_ID = "gsm8k:qsha256:b776409e5bbd4d63f2d0b3d630d21ac8ec56e70f84444a1f592835584d76c44d"

# Curated from source wording/rationale alone; never from model predictions.
SOURCE_EXCLUSIONS = {
    TADPOLES_ID: "Ambiguous source question.",
    "gsm8k:qsha256:08583e67384e90c9f4719350825f76e606a0a913c44aa8afa13d44a5dc1379df": (
        "Monthly totals are derived from weekly rates using an unstated four-week month."
    ),
    "gsm8k:qsha256:aa8117eb2f6758984419d06e19a0107dab523e49ea64df822b539a045e63ed0e": (
        "The phrase '3 times more' has multiple plausible mathematical readings."
    ),
    "gsm8k:qsha256:b7f59158b17ad0c29c87d8a94177e34cffceba9fe3ea77e028e09b79d6e4c2b9": (
        "The worked solution misapplies the stated 1.5 multiplier and yields an "
        "inconsistent average."
    ),
    "gsm8k:qsha256:b8c462eaeb0b0329edfd68eb6c3cf6b2cb0e3a187873cc33762360965072b073": (
        "The source categorizes seahorses as fish, making the requested category inconsistent."
    ),
    "gsm8k:qsha256:48de513784221b58d0ecb3aaa4e1afa7521b8d49c478555982bab76e13114e96": (
        "'2.5 times faster' is ambiguous between a speed ratio and an additive increase."
    ),
    "gsm8k:qsha256:d362889745c00cc131020ce61c3cc48a6b09d9d48f6750d903ef1409d29d9950": (
        "The source describes a speed improvement while the rationale applies a time reduction."
    ),
    "gsm8k:qsha256:82fc37f3955094866cdea7c8b282971f245eb872212aff8fb32bf16ce6c1fc60": (
        "Monthly interest does not specify simple versus compounded interest."
    ),
    "gsm8k:qsha256:952fe1fef538ba98ca878ece228231b0fa00bd3d384a98d9d0c5a71ead467ac0": (
        "Energy-consumption and billing units are stated inconsistently."
    ),
    "gsm8k:qsha256:8f2ecd574d8c8738ec87b6ba3a3af24f9cbf35d832733c98e65546f7ee59725d": (
        "The named business owner differs between the question (Kim) and target (Lucas)."
    ),
    "gsm8k:qsha256:055e011172b610fae76c080090cb1551109c2c66429e0a7537b05eee7781ebcb": (
        "The source says speed is doubled but the rationale doubles time per mile."
    ),
}


def _digest(text: bytes) -> str:
    return hashlib.sha256(text).hexdigest()


def _jsonl(rows: Sequence[Record]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True)
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
        + "\n"
        for row in rows
    ).encode("utf-8")


def select_questions(
    official_test_rows: Sequence[Record], *, seed: int = 42, size: int = 100
) -> list[Record]:
    """Select by a stable seeded hash, filtering curated exclusions before ranking."""
    if type(seed) is not int or seed < 0 or type(size) is not int or size < 1:
        raise ValueError("seed must be nonnegative and size must be positive")
    candidates = [
        row for row in official_test_rows if row["original_problem_id"] not in SOURCE_EXCLUSIONS
    ]
    if len({row["original_problem_id"] for row in candidates}) != len(candidates):
        raise ValueError("Official test source contains duplicate canonical IDs")
    ranked = sorted(
        candidates,
        key=lambda row: (
            _digest(f"{seed}:{SELECTION_PURPOSE}:{row['original_problem_id']}".encode()),
            row["original_problem_id"],
        ),
    )
    if len(ranked) < size:
        raise ValueError(f"Need {size} eligible official test questions; found {len(ranked)}")
    selected = ranked[:size]
    if any(row.get("source_split") != "test" or row.get("split") != "test" for row in selected):
        raise ValueError("Phase 14 selection must preserve the official test split")
    return selected


def build_variant_records(questions: Sequence[Record]) -> list[Record]:
    """Create originals and explicit translation-pending slots, never fake translations."""
    rows: list[Record] = []
    for question in questions:
        for language in LANGUAGES:
            is_english = language == "en"
            translated = question["original_question"] if is_english else ""
            flags = [] if is_english else ["translation_not_generated", "missing_text"]
            rows.append(
                {
                    "original_problem_id": question["original_problem_id"],
                    "source_split": question["source_split"],
                    "split": "test",
                    "language": language,
                    "source_text": question["original_question"],
                    "translated_text": translated,
                    "gold_answer": question["gold_answer"],
                    "translation_provider": "identity" if is_english else None,
                    "translation_model": "original" if is_english else None,
                    "translation_prompt_version": "identity-v1" if is_english else None,
                    "translation_timestamp": None,
                    "qa_flags": flags,
                    "qa_status": "source_original" if is_english else "needs_translation",
                    "approval_status": "not_required" if is_english else "pending",
                }
            )
    duplicate_flags = duplicate_translation_flags(rows)
    for row in rows:
        flag = duplicate_flags.get((row["original_problem_id"], row["language"]))
        if flag:
            row["qa_flags"].append(flag)
    return rows


def validate_variant_records(
    questions: Sequence[Record],
    variants: Sequence[Record],
    *,
    training_ids: set[str],
    validation_ids: set[str],
) -> dict[str, Any]:
    expected_ids = {row["original_problem_id"] for row in questions}
    pairs = [(row["original_problem_id"], row["language"]) for row in variants]
    if len(pairs) != len(set(pairs)):
        raise ValueError("Duplicate problem-language variant")
    by_language = {
        lang: {row["original_problem_id"] for row in variants if row["language"] == lang}
        for lang in LANGUAGES
    }
    if set(by_language) != {row["language"] for row in variants}:
        raise ValueError("Unexpected language variant")
    if any(ids != expected_ids for ids in by_language.values()):
        raise ValueError("Languages do not contain identical canonical IDs")
    training_overlap = expected_ids & training_ids
    validation_overlap = expected_ids & validation_ids
    if training_overlap or validation_overlap:
        raise ValueError("Evaluation IDs overlap training or validation IDs")
    source_by_id = {row["original_problem_id"]: row for row in questions}
    for row in variants:
        source = source_by_id[row["original_problem_id"]]
        if row["gold_answer"] != source["gold_answer"]:
            raise ValueError("Gold answer differs across language variants")
        if row["source_text"] != source["original_question"]:
            raise ValueError("Variant does not preserve exact source question")
        if row["language"] == "en" and row["translated_text"] != source["original_question"]:
            raise ValueError("English must be the unmodified source question")
        if row["language"] != "en" and row["approval_status"] not in ("pending", "approved"):
            raise ValueError("Invalid non-English approval status")
        if (
            row["language"] != "en"
            and row["approval_status"] == "approved"
            and not row["translated_text"].strip()
        ):
            raise ValueError("Cannot approve a missing translation")
    missing = sum(
        row["language"] != "en" and not row["translated_text"].strip() for row in variants
    )
    approved = sum(
        row["language"] != "en" and row["approval_status"] == "approved" for row in variants
    )
    return {
        "selected_questions": len(questions),
        "variants_by_language": {language: len(ids) for language, ids in by_language.items()},
        "unique_problem_language_pairs": len(pairs),
        "training_overlap": len(training_overlap),
        "validation_overlap": len(validation_overlap),
        "missing_non_english_translations": missing,
        "approved_non_english_translations": approved,
        "research_ready": (
            missing == 0
            and approved == 3 * len(questions)
            and len(training_overlap) == 0
            and len(validation_overlap) == 0
        ),
    }


def _human_review_csv(variants: Sequence[Record]) -> bytes:
    fields = (
        "canonical_problem_id",
        "english_original",
        "translated_question",
        "language",
        "gold_answer",
        "automated_quality_check_results",
        "translation_accuracy",
        "mathematical_equivalence",
        "ambiguity",
        "reviewer_comments",
        "approval_status",
    )
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in variants:
        if row["language"] == "en":
            continue
        writer.writerow(
            {
                "canonical_problem_id": row["original_problem_id"],
                "english_original": row["source_text"],
                "translated_question": row["translated_text"],
                "language": row["language"],
                "gold_answer": row["gold_answer"],
                "automated_quality_check_results": ";".join(row["qa_flags"]),
                "translation_accuracy": "pending",
                "mathematical_equivalence": "pending",
                "ambiguity": "pending",
                "reviewer_comments": "",
                "approval_status": "pending",
            }
        )
    return stream.getvalue().encode("utf-8")


def export_phase14_draft(
    output_dir: Path,
    *,
    questions: Sequence[Record],
    variants: Sequence[Record],
    source_manifest: Mapping[str, Any],
    source_manifest_sha256: str,
    train_ids: set[str],
    validation_ids: set[str],
    seed: int = 42,
    translation_blocker: str,
) -> Path:
    """Export an honest non-ready draft without altering source or older results."""
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    integrity = validate_variant_records(
        questions, variants, training_ids=train_ids, validation_ids=validation_ids
    )
    report = {
        **integrity,
        "automated_checks_are_heuristics": True,
        "flag_counts": dict(Counter(flag for row in variants for flag in row["qa_flags"])),
        "answer_leakage_checks": "not run: translated text is unavailable",
        "unit_currency_and_relation_checks": "not run: translated text is unavailable",
    }
    output_dir.mkdir(parents=True)
    files = {
        "selected_questions.jsonl": _jsonl(questions),
        "variants.jsonl": _jsonl(variants),
        "human_review.csv": _human_review_csv(variants),
        "qa_report.json": (
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode(),
    }
    artifacts = {}
    record_counts = {
        "selected_questions.jsonl": len(questions),
        "variants.jsonl": len(variants),
        "human_review.csv": sum(row["language"] != "en" for row in variants),
        "qa_report.json": None,
    }
    for name, content in files.items():
        (output_dir / name).write_bytes(content)
        artifact = {"path": name, "sha256": _digest(content)}
        if record_counts[name] is not None:
            artifact["count"] = record_counts[name]
        artifacts[name] = artifact
    manifest = {
        "schema_version": 1,
        "dataset_version": DATASET_VERSION,
        "status": "translation_blocked",
        "complete": False,
        "research_ready": False,
        "selection": {
            "source_dataset": "openai/gsm8k",
            "configuration": "main",
            "source_revision": source_manifest["dataset"]["resolved_revision"],
            "source_official_test_sha256": source_manifest["dataset"]["source_sha256"]["test"],
            "canonical_test_artifact_sha256": source_manifest["splits"]["test"]["sha256"],
            "selection_seed": seed,
            "algorithm": (
                f"SHA256(seed:{SELECTION_PURPOSE}:canonical_id), ascending, "
                "after curated exclusions"
            ),
            "selected_question_count": len(questions),
            "problem_ids": [row["original_problem_id"] for row in questions],
            "excluded_problem_ids_and_reasons": SOURCE_EXCLUSIONS,
            "training_overlap": integrity["training_overlap"],
            "validation_overlap": integrity["validation_overlap"],
            "based_on_model_correctness": False,
        },
        "translation": {
            "status": "not_generated",
            "translated_non_english_count": 0,
            "required_non_english_count": 3 * len(questions),
            "human_approved_count": 0,
            "provider": None,
            "blocker": translation_blocker,
            "paid_api_requests_made": 0,
            "mock_translations_used": 0,
        },
        "source_manifest_sha256": source_manifest_sha256,
        "counts": integrity,
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path
