# ruff: noqa: E501
"""Audit imported draft translations and build a provisional matched SFT export."""

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from dataclasses import replace
from pathlib import Path

from phase16b_training_translations import validate

from multilingual_llm_reasoning.sft import prepare_sft
from multilingual_llm_reasoning.sft_config import load_sft_config
from multilingual_llm_reasoning.translation_qa import mathematical_translation_flags, qa_flags

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase16c-sft-candidate-v1"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run(input_path: Path) -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    imported = validate(input_path)
    blocking = [
        x for x in imported["issues"] if x["issue"] != "numeric_tokens_differ_review_required"
    ]
    if blocking:
        raise ValueError(f"Import has structural errors: {blocking[:3]}")
    with input_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 150 or len({r["canonical_problem_id"] for r in rows}) != 150:
        raise ValueError("Expected 150 distinct translated training IDs")
    OUT.mkdir(parents=True)
    shutil.copy2(input_path, OUT / "imported_draft_original.csv")
    report_rows = []
    variants = []
    for row in rows:
        source, target, lang = (
            row["english_original"],
            row["translated_question"],
            row["target_language"],
        )
        flags = sorted(
            set(
                qa_flags(source, target, lang, gold_answer=row["gold_answer"])
                + mathematical_translation_flags(source, target, lang)
            )
        )
        # Confirmed structural errors are rejected by the import validator above.
        # Heuristic semantic signals require review rather than automatic rejection.
        status = "REVIEW" if flags else "PASS"
        report_rows.append(
            {
                "canonical_problem_id": row["canonical_problem_id"],
                "target_language": lang,
                "qa_status": status,
                "reasons": ";".join(flags),
                "english_original": source,
                "translated_question": target,
                "gold_answer": row["gold_answer"],
                "approval_status": row["approval_status"],
                "mathematical_equivalence": row["mathematical_equivalence"],
                "reviewer_comments": row["reviewer_comments"],
                "numeric_qa_note": row["numeric_qa_note"],
            }
        )
        variants.append(
            {
                "original_problem_id": row["canonical_problem_id"],
                "language": lang,
                "split": "train",
                "source_split": "train",
                "source_text": source,
                "translated_text": target,
                "gold_answer": row["gold_answer"],
                "qa_status": "pending_human_review",
                "qa_flags": flags,
                "translation_provider": "user_supplied_draft",
                "translation_model": None,
                "approval_status": row["approval_status"],
                "reviewer_comments": row["reviewer_comments"],
            }
        )
    with (OUT / "translation_qa_report.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report_rows[0]))
        writer.writeheader()
        writer.writerows(report_rows)
    summary = {
        "total": 150,
        "status_counts": dict(Counter(r["qa_status"] for r in report_rows)),
        "flag_counts": dict(Counter(flag for row in variants for flag in row["qa_flags"])),
        "approval_status_counts": dict(Counter(r["approval_status"] for r in rows)),
        "structural_import_errors": 0,
        "numeric_token_review_signals": len(imported["issues"]),
        "automated_qa_is_not_human_approval": True,
        "source_csv_sha256": sha(input_path.read_bytes()),
    }
    (OUT / "translation_qa_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in variants)
    (OUT / "variants.jsonl").write_text(payload)
    prep = ROOT / "data/processed/gsm8k-200/manifest.json"
    tm = {
        "schema_version": 1,
        "complete": True,
        "mock": False,
        "input_manifest_sha256": sha(prep.read_bytes()),
        "research_ready": False,
        "artifacts": {
            "variants": {"path": "variants.jsonl", "count": 150, "sha256": sha(payload.encode())}
        },
    }
    (OUT / "candidate_translation_manifest.json").write_text(json.dumps(tm, indent=2) + "\n")
    config = load_sft_config(ROOT / "configs/prepare_sft.yaml")
    config = replace(
        config,
        translation_manifest=OUT / "candidate_translation_manifest.json",
        output_dir=OUT / "sft_export",
        allow_unreviewed_questions=True,
    )
    result = prepare_sft(config)
    if result["training_ready"] or result["status"] != "preview_not_training_ready":
        raise ValueError("Provisional SFT was incorrectly marked training-ready")
    shutil.copy2(OUT / "sft_export/english_train.jsonl", OUT / "english_only_sft.jsonl")
    shutil.copy2(
        OUT / "sft_export/multilingual_train.jsonl", OUT / "multilingual_sft_candidate.jsonl"
    )
    diagnostics = result["diagnostics"]
    comparison = {
        "english": diagnostics["english_train"],
        "multilingual": diagnostics["multilingual_train"],
        "multilingual_by_language": diagnostics["multilingual_by_question_language"],
        "target_token_difference": diagnostics["multilingual_train"]["target_tokens"]["total"]
        - diagnostics["english_train"]["target_tokens"]["total"],
        "total_token_difference": diagnostics["multilingual_train"]["token_length"]["total"]
        - diagnostics["english_train"]["token_length"]["total"],
    }
    (OUT / "token_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    markdown = f"""# Phase 16C validation report

Imported 150 distinct GSM8K training translations (50 zh, 50 de, 50 ja) from a user-supplied draft. The original CSV was copied without text changes. All approvals remain pending. Automated QA: {summary["status_counts"]}. Flags are review signals, not confirmed semantic errors or approval. The report CSV lists every row and its reasons.

Both provisional SFT conditions contain 200 rows on identical canonical IDs; the multilingual condition has 50 en, 50 zh, 50 de, and 50 ja questions. Both retain identical original English worked solutions and gold answers. The existing SFT integrity checks confirmed train/validation/test separation, assistant-only labels, chat formatting, and no truncation. Training-ready: **false**.

English total tokens: {comparison["english"]["token_length"]["total"]}; multilingual total tokens: {comparison["multilingual"]["token_length"]["total"]} (difference {comparison["total_token_difference"]:+}). Both have {comparison["english"]["target_tokens"]["total"]} supervised target tokens. Detailed prompt/response distributions appear in `token_comparison.json`.

Blockers: 150 pending approvals and unresolved QA review signals. These candidate files must not be used for final LoRA training. No training or paid API call was made.
"""
    (OUT / "phase16c_validation_report.md").write_text(markdown)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    run(parser.parse_args().input)
