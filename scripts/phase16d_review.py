"""Recheck Phase 16C heuristics and prepare a prioritized translation review queue."""

import csv
import json
from collections import Counter
from pathlib import Path

from multilingual_llm_reasoning.translation_qa import mathematical_translation_flags, qa_flags

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/phase16c-sft-candidate-v1"
OUT = ROOT / "outputs/phase16d-translation-review-v1"

FIELDS = [
    "review_order",
    "severity",
    "canonical_problem_id",
    "target_language",
    "english_original",
    "current_translation",
    "gold_answer",
    "qa_flags",
    "suspected_issue",
    "suggested_correction",
    "english_worked_solution",
    "source_sha256",
    "existing_reviewer_comments",
    "approval_status",
    "reviewer_decision",
    "final_corrected_translation",
    "new_reviewer_comments",
]


def severity(flags: list[str]) -> tuple[int, str, str]:
    if "possible_answer_leakage" in flags or "possible_solution_leakage" in flags:
        return (
            0,
            "P0 — possible answer leakage",
            "Check whether the draft exposes the numerical solution.",
        )
    if "missing_final_question" in flags:
        return (
            1,
            "P1 — possible missing final question",
            "Check that the last question is complete.",
        )
    if any(
        x in flags
        for x in (
            "missing_numeric_values",
            "duplicated_or_added_numeric_values",
            "missing_rounding_instruction",
            "changed_rounding_direction",
            "missing_percentage_reference",
            "missing_currency_reference",
            "unclear_each_quantity_relation",
        )
    ):
        return (
            2,
            "P2 — mathematical content",
            "Compare quantities, units, conditions, and operations with the source.",
        )
    if any(
        x in flags
        for x in (
            "relational_cues_not_detected",
            "possibly_incomplete_sentences",
            "possibly_missing_text",
        )
    ):
        return (
            3,
            "P3 — semantic completeness",
            "Check relationships and completeness; cue absence alone is not an error.",
        )
    if flags:
        return (
            4,
            "P4 — formatting or tokenization",
            "Check localized number and unit wording; token mismatch alone is not an error.",
        )
    return 5, "P5 — standard review", "No automated issue detected; verify semantic equivalence."


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    with (SOURCE / "imported_draft_original.csv").open(encoding="utf-8-sig", newline="") as handle:
        original = list(csv.DictReader(handle))
    with (SOURCE / "translation_qa_report.csv").open(encoding="utf-8-sig", newline="") as handle:
        old = list(csv.DictReader(handle))
    candidate = [
        json.loads(line)
        for line in (SOURCE / "multilingual_sft_candidate.jsonl").read_text().splitlines()
    ]
    candidate_by_key = {(r["original_problem_id"], r["question_language"]): r for r in candidate}
    old_by_key = {(r["canonical_problem_id"], r["target_language"]): r for r in old}
    if len(original) != 150 or len(old_by_key) != 150:
        raise ValueError("Expected 150 unique review rows")
    prepared = []
    removed_final_alerts = 0
    for row in original:
        key = (row["canonical_problem_id"], row["target_language"])
        if (
            row["approval_status"] != "pending"
            or candidate_by_key[key]["question_text"] != row["translated_question"]
        ):
            raise ValueError("Candidate differs from pending source draft")
        flags = sorted(
            set(
                qa_flags(
                    row["english_original"],
                    row["translated_question"],
                    key[1],
                    gold_answer=row["gold_answer"],
                )
                + mathematical_translation_flags(
                    row["english_original"], row["translated_question"], key[1]
                )
            )
        )
        if (
            "missing_final_question" in old_by_key[key]["reasons"]
            and "missing_final_question" not in flags
        ):
            removed_final_alerts += 1
        rank, label, explanation = severity(flags)
        prepared.append(
            (
                rank,
                {
                    "review_order": "",
                    "severity": label,
                    "canonical_problem_id": key[0],
                    "target_language": key[1],
                    "english_original": row["english_original"],
                    "current_translation": row["translated_question"],
                    "gold_answer": row["gold_answer"],
                    "qa_flags": ";".join(flags),
                    "suspected_issue": explanation,
                    "suggested_correction": "",
                    "english_worked_solution": row["english_worked_solution"],
                    "source_sha256": row["source_sha256"],
                    "existing_reviewer_comments": row["reviewer_comments"],
                    "approval_status": "pending",
                    "reviewer_decision": "",
                    "final_corrected_translation": "",
                    "new_reviewer_comments": "",
                },
            )
        )
    prepared.sort(
        key=lambda item: (item[0], item[1]["canonical_problem_id"], item[1]["target_language"])
    )
    rows = [r for _, r in prepared]
    for index, row in enumerate(rows, 1):
        row["review_order"] = index
    OUT.mkdir(parents=True)
    write_csv(OUT / "review_queue.csv", rows)
    for index in range(6):
        write_csv(OUT / f"review_batch_{index + 1:02d}.csv", rows[index * 25 : (index + 1) * 25])
    counts = Counter(row["severity"] for row in rows)
    flags = Counter(flag for row in rows for flag in row["qa_flags"].split(";") if flag)
    old_counts = Counter(flag for row in old for flag in row["reasons"].split(";") if flag)
    summary = {
        "rows": len(rows),
        "severity_counts": dict(counts),
        "flag_counts": dict(flags),
        "prior_flag_counts": dict(old_counts),
        "deterministically_resolved_final_question_alerts": removed_final_alerts,
        "remaining_missing_final_question_alerts": flags["missing_final_question"],
        "confirmed_mathematical_inconsistencies": 0,
        "semantic_review_required": len(rows),
        "note": "All 150 require an explicit human decision; automated PASS is not approval.",
    }
    (OUT / "review_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    text = f"""# Phase 16D translation QA analysis

The original 150 drafts and all pending approvals are unchanged. This queue is
sorted by suspected severity and includes all 150 rows, including those with
no automated flags. Review batches contain exactly 25 rows each.

Severity counts: {dict(counts)}. Refreshed flag counts: {dict(flags)}.
A possible answer-leakage signal is first in the queue. Automated flags are
review heuristics, not confirmed translation errors.

The previous missing-final-question detector flagged 25 Japanese translations
ending in the valid question form `か。`. Direct deterministic inspection found
that all 25 end with that form. The detector now recognizes it, removing
{removed_final_alerts} false alerts. A regression test retains detection of a
Japanese statement that lacks a question. This resolves only the
punctuation/question-form alert; it does not prove that the translated question
has the same meaning.

Raw numeric and currency token differences are often due to localized digits,
number words, or units. These cannot establish equivalence. Missing/added
quantities, changed relationships, rounding, potential leakage, and ambiguity
require semantic review. No mathematical inconsistency has been confirmed by
deterministic checks. Suggested corrections remain blank because no replacement
wording is justified without review. All reviewer decisions and new comments
are blank; prior comments are preserved separately.

The candidate SFT remains provisional and must not be used for LoRA training
until every translation has an explicit review decision and unresolved critical
issues are resolved.
"""
    (OUT / "phase16d_qa_analysis.md").write_text(text)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run()
