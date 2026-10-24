"""Paired language-level analysis for a single four-language baseline condition."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .baseline_scoring import score_record
from .numeric import normalize_numeric
from .scoring import PARSER_VERSION
from .translation import LANGUAGES


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze_baseline(
    predictions_path: Path,
    test_set_path: Path,
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Validate and summarize one real four-language baseline without dropping rows."""
    if output_dir.exists():
        raise ValueError("Analysis output exists; choose a new directory")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete") or not manifest.get("research_ready") or manifest.get("mock"):
        raise ValueError("Analysis requires a complete, research-ready, non-mock dataset")
    artifact = manifest.get("artifacts", {}).get("variants", {})
    artifact_path = (manifest_path.parent / artifact.get("path", "")).resolve()
    if artifact_path != test_set_path.resolve() or _sha256(test_set_path) != artifact.get("sha256"):
        raise ValueError("Test set does not match the research-ready manifest")

    expected_rows = _read_jsonl(test_set_path)
    raw_rows = _read_jsonl(predictions_path)
    if not raw_rows:
        raise ValueError("Predictions are empty")
    for key in (
        "backend",
        "model_checkpoint",
        "model_revision",
        "prompt_template_version",
        "chat_template_sha256",
        "decoding",
        "parser_version",
    ):
        if len({json.dumps(row.get(key), sort_keys=True) for row in raw_rows}) != 1:
            raise ValueError(f"Baseline prediction settings differ across rows: {key}")
    if raw_rows[0].get("parser_version") != PARSER_VERSION:
        raise ValueError("Predictions do not record the current parser policy")
    expected: dict[tuple[str, str], dict[str, Any]] = {}
    for row in expected_rows:
        key = (row.get("original_problem_id"), row.get("language"))
        if key[1] not in LANGUAGES or not isinstance(key[0], str) or key in expected:
            raise ValueError("Test set has an invalid or duplicate problem/language key")
        expected[key] = row
    ids = sorted({identity for identity, _ in expected})
    if not ids or set(expected) != {(identity, lang) for identity in ids for lang in LANGUAGES}:
        raise ValueError("Test set must contain identical complete IDs across four languages")

    joined: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in raw_rows:
        key = (raw.get("original_problem_id"), raw.get("language"))
        if key not in expected or key in joined:
            raise ValueError("Predictions contain an unexpected or duplicate problem/language row")
        if raw.get("backend") != "transformers" or raw.get("mock_model") is True:
            raise ValueError("Real baseline analysis refuses mock or non-Transformers outputs")
        if normalize_numeric(raw.get("gold_answer")) != normalize_numeric(
            expected[key].get("gold_answer")
        ):
            raise ValueError("Prediction gold answer differs from the approved test set")
        joined[key] = score_record(raw)
    if set(joined) != set(expected):
        raise ValueError("Predictions are incomplete; no inner-join shrinking is allowed")

    per_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (_, language), row in joined.items():
        per_language[language].append(row)
    language_metrics = {}
    for language in LANGUAGES:
        rows = per_language[language]
        total = len(rows)
        correct = sum(bool(row["correct"]) for row in rows)
        parsed = sum(row["parse_status"] == "ok" for row in rows)
        parse_failures = sum(
            row["parse_status"] not in ("ok", "pending", "inference_error", "invalid_gold")
            for row in rows
        )
        language_metrics[language] = {
            "n_expected": total,
            "correct": correct,
            "accuracy": correct / total,
            "parsed": parsed,
            "parse_failures": parse_failures,
            "parse_failure_rate": parse_failures / total,
            "inference_failures_or_pending": sum(
                row["parse_status"] in ("pending", "inference_error") for row in rows
            ),
            "parse_status_counts": {
                status: sum(row["parse_status"] == status for row in rows)
                for status in sorted({row["parse_status"] for row in rows})
            },
        }

    gaps = {
        language: language_metrics["en"]["accuracy"] - language_metrics[language]["accuracy"]
        for language in LANGUAGES
        if language != "en"
    }
    question_rows = []
    disagreements = []
    all_correct = 0
    all_same_answer = 0
    unanimous_wrong = []
    for identity in ids:
        outcomes = {}
        for language in LANGUAGES:
            row = joined[identity, language]
            outcomes[language] = {
                "gold_answer": row["gold_answer"],
                "prediction": row.get("prediction"),
                "parsed_answer": row.get("parsed_answer"),
                "parse_status": row.get("parse_status"),
                "correct": bool(row.get("correct")),
            }
        correct_values = {item["correct"] for item in outcomes.values()}
        answer_values = {
            (item["parse_status"], item["parsed_answer"]) for item in outcomes.values()
        }
        answer_agreement = len(answer_values) == 1 and all(
            item["parse_status"] == "ok" for item in outcomes.values()
        )
        all_languages_correct = all(item["correct"] for item in outcomes.values())
        all_correct += int(all_languages_correct)
        all_same_answer += int(answer_agreement)
        if correct_values == {False}:
            unanimous_wrong.append(identity)
        record = {
            "original_problem_id": identity,
            "gold_answer": expected[identity, "en"]["gold_answer"],
            "all_languages_correct": all_languages_correct,
            "same_parsed_answer_all_languages": answer_agreement,
            "correctness_disagreement": len(correct_values) > 1,
            "answer_or_parse_disagreement": len(answer_values) > 1,
            "outcomes": outcomes,
        }
        question_rows.append(record)
        if record["correctness_disagreement"] or record["answer_or_parse_disagreement"]:
            disagreements.append(record)

    n_questions = len(ids)
    metrics = {
        "schema_version": 1,
        "analysis_type": "single-condition-cross-lingual-baseline",
        "condition_id": "base",
        "model_checkpoint": raw_rows[0].get("model_checkpoint"),
        "model_revision": raw_rows[0].get("model_revision"),
        "parser_version": PARSER_VERSION,
        "n_originals": n_questions,
        "n_predictions": len(joined),
        "preliminary": True,
        "sample_size_note": f"Preliminary results: only {n_questions} original questions.",
        "language_metrics": language_metrics,
        "english_minus_other_language_accuracy_gaps": gaps,
        "cross_language_consistency": {
            "all_four_correct": all_correct / n_questions,
            "all_four_correct_count": all_correct,
            "same_parsed_answer": all_same_answer / n_questions,
            "same_parsed_answer_count": all_same_answer,
            "n_originals": n_questions,
        },
        "unanimously_wrong_problem_ids": unanimous_wrong,
        "disagreement_count": len(disagreements),
        "denominator_policy": (
            "All expected problem-language rows remain in accuracy and parse-failure denominators."
        ),
        "mock": False,
        "sources": {
            "predictions": {
                "path": str(predictions_path.resolve()),
                "sha256": _sha256(predictions_path),
            },
            "test_set": {"path": str(test_set_path.resolve()), "sha256": _sha256(test_set_path)},
            "dataset_manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": _sha256(manifest_path),
            },
        },
    }
    output_dir.mkdir(parents=True)
    (output_dir / "summary.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for filename, rows in (
        ("question_outcomes.jsonl", question_rows),
        ("disagreements.jsonl", disagreements),
    ):
        (output_dir / filename).write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    with (output_dir / "language_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["language", *language_metrics["en"].keys()])
        writer.writeheader()
        for language in LANGUAGES:
            writer.writerow({"language": language, **language_metrics[language]})
    with (output_dir / "disagreements.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "original_problem_id",
                "gold_answer",
                "correctness_disagreement",
                "answer_or_parse_disagreement",
                "outcomes",
            ],
        )
        writer.writeheader()
        for row in disagreements:
            writer.writerow(
                {
                    "original_problem_id": row["original_problem_id"],
                    "gold_answer": row["gold_answer"],
                    "correctness_disagreement": row["correctness_disagreement"],
                    "answer_or_parse_disagreement": row["answer_or_parse_disagreement"],
                    "outcomes": json.dumps(row["outcomes"], ensure_ascii=False),
                }
            )
    lines = [
        "# Real multilingual baseline analysis",
        "",
        f"**Preliminary:** {n_questions} held-out questions; no uncertainty interval is estimated.",
        "",
        "| Language | Correct / N | Accuracy | Parse failures / N | Parse-failure rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for language in LANGUAGES:
        row = language_metrics[language]
        lines.append(
            f"| {language} | {row['correct']} / {row['n_expected']} | {row['accuracy']:.1%} | "
            f"{row['parse_failures']} / {row['n_expected']} | {row['parse_failure_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            f"English-minus-other accuracy gaps: {json.dumps(gaps, sort_keys=True)} "
            "(positive means English higher).",
            f"All four correct: {all_correct}/{n_questions}; "
            f"same parsed answer in all four: {all_same_answer}/{n_questions}.",
            f"Question-level disagreements: {len(disagreements)}. "
            "Full outcomes are in `question_outcomes.jsonl`; "
            "the disagreement-only files are empty when no language outputs disagree.",
            f"Unanimously wrong IDs: {', '.join(unanimous_wrong) if unanimous_wrong else 'none'}.",
            "",
            "These results are a small-sample baseline only. They do not establish "
            "translation quality or a general language effect.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics
