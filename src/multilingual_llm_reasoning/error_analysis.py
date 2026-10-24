"""Reproducible matched error evidence; subjective categories remain human annotations."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from .baseline_scoring import score_record
from .comparison_metrics import CONDITIONS, paired_summary
from .translation import LANGUAGES

CATEGORIES = (
    "translation ambiguity",
    "comprehension",
    "arithmetic",
    "multi-step logic",
    "formatting/parser",
    "unknown",
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze_results(
    predictions: Path,
    test_set: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    bootstrap_samples: int = 5000,
) -> dict:
    if output_dir.exists():
        raise ValueError("Analysis output exists; choose a new directory to preserve human reviews")
    raw, expected = read_jsonl(predictions), read_jsonl(test_set)
    # Strict three-condition/four-language join; no dropping duplicates or incomplete keys.
    metrics = paired_summary(raw, expected, seed=seed, bootstrap_samples=bootstrap_samples)
    lookup = {(r["original_problem_id"], r["language"]): r for r in expected}
    scored = {
        (r["condition_id"], r["original_problem_id"], r["language"]): score_record(r) for r in raw
    }
    identities = sorted({key[0] for key in lookup})
    reasons = {key: set() for key in scored}
    events = []
    for identity in identities:
        for condition in CONDITIONS:
            group = [scored[condition, identity, lang] for lang in LANGUAGES]
            correctness_disagreement = len({r["correct"] for r in group}) > 1
            answer_disagreement = len({(r["parse_status"], r["parsed_answer"]) for r in group}) > 1
            if correctness_disagreement or answer_disagreement:
                events.append(
                    {
                        "event": "cross_language_disagreement",
                        "original_problem_id": identity,
                        "condition_id": condition,
                        "correctness_disagreement": correctness_disagreement,
                        "answer_or_parse_disagreement": answer_disagreement,
                        "outcomes": {
                            lang: {
                                k: scored[condition, identity, lang][k]
                                for k in ("correct", "parsed_answer", "parse_status")
                            }
                            for lang in LANGUAGES
                        },
                    }
                )
                for lang in LANGUAGES:
                    reasons[condition, identity, lang].add("cross_language_disagreement")
        for lang in LANGUAGES:
            variant = lookup[identity, lang]
            flags = variant.get("qa_flags", [])
            qa = variant.get("qa_status", "missing_qa_status")
            if flags or qa not in ("source_original", "human_approved"):
                events.append(
                    {
                        "event": "translation_flagged",
                        "original_problem_id": identity,
                        "language": lang,
                        "qa_status": qa,
                        "qa_flags": flags,
                        "review": variant.get("review"),
                    }
                )
                for condition in CONDITIONS:
                    reasons[condition, identity, lang].add("translation_flagged")
            for treatment, reference in (
                ("english_lora", "base"),
                ("multilingual_lora", "base"),
                ("multilingual_lora", "english_lora"),
            ):
                before, after = scored[reference, identity, lang], scored[treatment, identity, lang]
                if before["correct"] != after["correct"]:
                    direction = (
                        "incorrect_to_correct" if after["correct"] else "correct_to_incorrect"
                    )
                    events.append(
                        {
                            "event": direction,
                            "original_problem_id": identity,
                            "language": lang,
                            "reference": reference,
                            "condition_id": treatment,
                            "before_prediction": before["prediction"],
                            "after_prediction": after["prediction"],
                            "before_parse_status": before["parse_status"],
                            "after_parse_status": after["parse_status"],
                        }
                    )
                    reasons[reference, identity, lang].add(direction)
                    reasons[treatment, identity, lang].add(direction)
    evidence, review = [], []
    for key in sorted(scored):
        condition, identity, lang = key
        row, source = scored[key], lookup[identity, lang]
        if not row["correct"]:
            reasons[key].add("incorrect_or_unscored")
        if not reasons[key]:
            continue
        review_id = hashlib.sha256(":".join(key).encode()).hexdigest()[:20]
        record = {
            "review_id": review_id,
            "original_problem_id": identity,
            "condition_id": condition,
            "language": lang,
            "original_question": source.get("source_text", source.get("original_question")),
            "translated_question": source.get("translated_text"),
            "gold_answer": row["gold_answer"],
            "prediction": row.get("prediction"),
            "parsed_answer": row["parsed_answer"],
            "correct": row["correct"],
            "parse_status": row["parse_status"],
            "inference_status": row.get("inference_status", "completed"),
            "qa_status": source.get("qa_status"),
            "qa_flags": source.get("qa_flags", []),
            "translation_review": source.get("review"),
            "reasons": sorted(reasons[key]),
            "model_checkpoint": row.get("model_checkpoint"),
            "model_revision": row.get("model_revision"),
            "adapter_audit": row.get("adapter_audit"),
        }
        evidence.append(record)
        review.append(
            {
                k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                for k, v in {
                    **record,
                    "category_options": list(CATEGORIES),
                    "reviewer_category": "",
                    "reviewer_notes": "",
                    "reviewer": "",
                }.items()
            }
        )
    report = {
        "schema_version": 1,
        "n_originals": len(identities),
        "n_predictions": len(raw),
        "n_review_rows": len(review),
        "event_counts": dict(sorted(Counter(e["event"] for e in events).items())),
        "affected_original_ids": sorted({r["original_problem_id"] for r in evidence}),
        "sources": {
            "predictions": {"path": str(predictions.resolve()), "sha256": file_hash(predictions)},
            "test_set": {"path": str(test_set.resolve()), "sha256": file_hash(test_set)},
        },
        "mock": any(r.get("backend") == "mock" for r in raw),
        "category_options": list(CATEGORIES),
        "subjective_categories_assigned": False,
        "method": (
            "Re-score raw outputs; exact paired join; all matching cases exported "
            "without sampling. Flags are review signals, not translation-error ground "
            "truth."
        ),
        "bootstrap": metrics["bootstrap"],
    }
    output_dir.mkdir(parents=True)
    for name, values in (("events", events), ("evidence", evidence)):
        (output_dir / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in values)
        )
    fields = (
        list(review[0])
        if review
        else [
            "review_id",
            "original_problem_id",
            "condition_id",
            "language",
            "category_options",
            "reviewer_category",
            "reviewer_notes",
            "reviewer",
        ]
    )
    write_csv(output_dir / "human_review.csv", review, fields)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    counts = (
        "\n".join(f"- {name}: {count}" for name, count in report["event_counts"].items())
        or "- No events."
    )
    ids = "\n".join(f"- `{identity}`" for identity in report["affected_original_ids"]) or "- None."
    status_note = (
        "**MOCK RESULTS — pipeline diagnostics only.**"
        if report["mock"]
        else "Saved predictions; inspect run provenance before interpreting results."
    )
    text = f"""# Matched error-analysis report

{status_note}

{len(identities)} original questions; {len(raw)} predictions; {len(review)} review rows.
No subjective categories were assigned automatically. Correctness changes may be caused
by parsing or inference failures, not reasoning changes. Translation QA flags are
review signals, not established translation errors. Unflagged translations can still
be wrong; numeric-token equality does not prove semantic correctness.

## Matched event counts

{counts}

Events can overlap. Full matched before/after evidence is in [events.jsonl](events.jsonl).
Each row in [human_review.csv](human_review.csv) links by review ID to
[evidence.jsonl](evidence.jsonl), with original ID, source question, translation,
raw output, gold, parse status and provenance. Fill reviewer_category, reviewer_notes,
and reviewer; category_options lists suggestions only. Preserve the raw JSON evidence.
Source file hashes and method are recorded in [report.json](report.json).

## Affected original question IDs

{ids}
"""
    (output_dir / "report.md").write_text(text)
    return report
