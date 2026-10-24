"""Recompute scores from raw predictions; keep failures in the primary denominator."""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .numeric import normalize_numeric
from .scoring import PARSER_VERSION, parse_final_answer

Record = dict[str, Any]


def score_record(row: Record) -> Record:
    result = {**row, "parsed_answer": None, "correct": False, "parser_version": PARSER_VERSION}
    try:
        gold = normalize_numeric(row.get("gold_answer"))
    except ValueError:
        result["parse_status"] = "invalid_gold"
        return result
    result["gold_answer"] = gold
    status = row.get("inference_status", "completed")
    if status != "completed":
        result["parse_status"] = "pending" if status == "pending" else "inference_error"
        return result
    parsed = parse_final_answer(row.get("prediction"))
    result.update(
        parsed_answer=parsed.answer,
        parse_status=parsed.status,
        correct=parsed.status == "ok" and parsed.answer == gold,
    )
    return result


def summarize_scores(rows: list[Record]) -> Record:
    def summary(group: list[Record]) -> Record:
        statuses = Counter(row["parse_status"] for row in group)
        valid = statuses["ok"]
        correct = sum(row["correct"] for row in group)
        inference_errors = statuses["inference_error"] + statuses["pending"]
        invalid = statuses["invalid_record"] + statuses["invalid_gold"]
        return {
            "total": len(group),
            "correct": correct,
            "accuracy": correct / len(group) if group else None,
            "parsed": valid,
            "parse_failures": len(group) - valid - inference_errors - invalid,
            "inference_failures_or_pending": inference_errors,
            "invalid_records": invalid,
            "parse_status_counts": dict(statuses),
            "accuracy_among_parsed": correct / valid if valid else None,
        }

    languages: dict[str, list[Record]] = defaultdict(list)
    for row in rows:
        languages[str(row.get("language", "unknown"))].append(row)
    return {
        "parser_version": PARSER_VERSION,
        "overall": summary(rows),
        "by_language": {lang: summary(group) for lang, group in sorted(languages.items())},
        "denominator_policy": "Primary accuracy includes parse failures, errors and pending rows.",
    }


def score_file(path: Path, output_dir: Path) -> Record:
    """Preserve every physical input line, including malformed JSON and duplicate records."""
    if output_dir.exists():
        raise FileExistsError("Scoring output directory already exists; choose a new path")
    rows = []
    seen = set()
    for line_number, raw_line in enumerate(path.read_bytes().splitlines(), 1):
        line = raw_line.decode("utf-8")
        error = None
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("expected a JSON object")
            for field in ("original_problem_id", "language", "model_checkpoint", "model_revision"):
                if not isinstance(row.get(field), str) or not row[field]:
                    # Failed initialization has no resolved model revision, but is still scored.
                    if (
                        field == "model_revision"
                        and row.get("inference_status", "completed") != "completed"
                    ):
                        continue
                    raise ValueError(f"missing/non-string {field}")
            if row.get("prediction") is not None and not isinstance(row["prediction"], str):
                raise ValueError("prediction must be a string or null")
            key = tuple(
                row.get(field)
                for field in (
                    "original_problem_id",
                    "language",
                    "model_checkpoint",
                    "model_revision",
                )
            )
            if key in seen:
                raise ValueError("duplicate model/problem/language record")
            seen.add(key)
        except (ValueError, TypeError) as exc:
            error = str(exc)
        if error:
            rows.append(
                {
                    "input_line": line_number,
                    "raw_line": line,
                    "error": error,
                    "parse_status": "invalid_record",
                    "parsed_answer": None,
                    "correct": False,
                }
            )
        else:
            rows.append({**score_record(row), "input_line": line_number})
    report = summarize_scores(rows)
    report["input_path"] = str(path.resolve())
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, selected in (
        ("scored_predictions.jsonl", rows),
        (
            "parse_failures.jsonl",
            [
                r
                for r in rows
                if r["parse_status"]
                not in ("ok", "inference_error", "pending", "invalid_record", "invalid_gold")
            ],
        ),
        (
            "score_errors.jsonl",
            [
                r
                for r in rows
                if r["parse_status"]
                in ("invalid_record", "invalid_gold", "inference_error", "pending")
            ],
        ),
    ):
        (output_dir / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
            encoding="utf-8",
        )
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
