"""Freeze the user-approved Phase 16 SFT pair without launching training."""

import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
PHASE16G = ROOT / "outputs/phase16g-final-dataset-candidate-v1"
PHASE16F = ROOT / "outputs/phase16f-review-reconciliation-v1"
OUT = ROOT / "outputs/phase16h-approved-sft-freeze-v1"
MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
MAX_LENGTH = 2048
SEED = 42

EXPECTED_DECISIONS = {
    "gsm8k:qsha256:19e6011ec9f82583a91f31d998b951617ed03a9ba629d566acda7e5a38bd20f5": "exclude",
    "gsm8k:qsha256:f76b8a5a42d5cc497fdde9c91aba1569a8cdede0255f860045e4319132f97087": "exclude",
    "gsm8k:qsha256:fb24638b2266e55947917a6fb92318e006476a893b57032f826692eae8e3cce9": "exclude",
    "gsm8k:qsha256:c980b7c245475f89c8b47fe2aa95324ffb99188aa271ce2773d421a067981a33": "retain",
}


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def jsonl_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def get_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def summarize(rows: list[dict]) -> dict:
    lengths = [row["token_length"] for row in rows]
    return {
        "examples": len(rows),
        "total_tokens": sum(lengths),
        "mean_tokens": sum(lengths) / len(lengths),
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "over_max_length": sum(n > MAX_LENGTH for n in lengths),
    }


def run() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    reviews = csv_rows(PHASE16G / "reviewed_translations_candidate.csv")
    proposals = csv_rows(PHASE16G / "proposed_exclusions.csv")
    if len(reviews) != 150 or len(proposals) != 4:
        raise ValueError("Phase 16G counts differ from reviewed 150 / four source decisions")
    review_keys = [(r["canonical_problem_id"], r["target_language"]) for r in reviews]
    if len(set(review_keys)) != 150 or Counter(k[1] for k in review_keys) != {
        "zh": 50,
        "de": 50,
        "ja": 50,
    }:
        raise ValueError("Reviewed translations contain duplicate or missing IDs")
    proposal_by_id = {r["canonical_problem_id"]: r for r in proposals}
    if set(proposal_by_id) != set(EXPECTED_DECISIONS):
        raise ValueError("Source decision canonical IDs differ from authorization")
    for identifier, action in EXPECTED_DECISIONS.items():
        expected = "propose_exclude" if action == "exclude" else "propose_retain"
        if proposal_by_id[identifier]["proposed_action"] != expected:
            raise ValueError(f"Source decision mapping differs for {identifier}")
    excluded = {
        identifier for identifier, action in EXPECTED_DECISIONS.items() if action == "exclude"
    }
    excluded_reviews = [r for r in reviews if r["canonical_problem_id"] in excluded]
    if (
        len(excluded_reviews) != 3
        or len({r["canonical_problem_id"] for r in excluded_reviews}) != 3
    ):
        raise ValueError("Three exclusions must correspond to exactly three translations")
    retained_reviews = [r for r in reviews if r["canonical_problem_id"] not in excluded]
    if len(retained_reviews) != 147:
        raise ValueError("Expected 147 retained translations")
    if any(r["approval_status"] != "pending" for r in reviews):
        raise ValueError("Phase 16G candidate unexpectedly contains prior approvals")

    # Confirm the two named translation decisions against the preserved change log.
    changes = csv_rows(PHASE16F / "phase16f_translation_change_log.csv")
    change_by_key = {(r["canonical_problem_id"], r["target_language"]): r for r in changes}
    review_by_key = {(r["canonical_problem_id"], r["target_language"]): r for r in reviews}
    if len(change_by_key) != 2 or Counter(r["action"] for r in changes) != {
        "revised": 1,
        "retained": 1,
    }:
        raise ValueError("German/Chinese change log differs")
    for key, change in change_by_key.items():
        if (
            key not in review_by_key
            or review_by_key[key]["current_translation"] != change["updated_translation"]
        ):
            raise ValueError(f"Translation change not reflected in candidate: {key}")
        if (
            change["action"] == "retained"
            and change["previous_translation"] != change["updated_translation"]
        ):
            raise ValueError("Chinese retained translation was modified")
        if change["action"] == "revised" and key[1] != "de":
            raise ValueError("Revised translation is not German")

    source = jsonl_rows(ROOT / "data/processed/gsm8k-200/subsets/train.jsonl")
    source_by_id = {r["original_problem_id"]: r for r in source}
    for row in reviews:
        identifier = row["canonical_problem_id"]
        original = source_by_id[identifier]
        if (
            row["english_original"] != original["original_question"]
            or row["english_worked_solution"] != original["original_answer"]
            or row["gold_answer"] != original["gold_answer"]
        ):
            raise ValueError(f"Original GSM8K text or answer differs: {identifier}")
        if hashlib.sha256(row["english_original"].encode()).hexdigest() != row["source_sha256"]:
            raise ValueError(f"Source hash differs: {identifier}")

    approval_rows = []
    for row in reviews:
        identifier = row["canonical_problem_id"]
        is_excluded = identifier in excluded
        approval_rows.append(
            {
                **row,
                "approval_status": "excluded" if is_excluded else "approved",
                "approval_source": "user_authorization",
                "review_method": "AI_assisted_semantic_review",
                "approval_scope": "Phase_16_experiment",
                "approval_provenance": (
                    "Explicit user authorization in Phase 16H after AI-assisted "
                    "semantic review; not independent expert verification."
                ),
                "exclusion_reason": proposal_by_id[identifier]["reason"] if is_excluded else "",
                "source_issue": proposal_by_id[identifier]["issue_category"]
                if identifier in proposal_by_id
                else "",
            }
        )
    if Counter(r["approval_status"] for r in approval_rows) != {"approved": 147, "excluded": 3}:
        raise ValueError("Approval count differs from authorization")

    english = jsonl_rows(PHASE16G / "english_only_sft_candidate.jsonl")
    multilingual = jsonl_rows(PHASE16G / "multilingual_sft_candidate.jsonl")
    if len(english) != 197 or len(multilingual) != 197:
        raise ValueError("Phase 16G candidate is not 197/197")
    eids = [r["original_problem_id"] for r in english]
    mids = [r["original_problem_id"] for r in multilingual]
    if len(set(eids)) != 197 or len(set(mids)) != 197 or set(eids) != set(mids):
        raise ValueError("Final condition IDs are not aligned")
    if set(eids) & excluded or set(eids) != {
        r["canonical_problem_id"] for r in retained_reviews
    } | {r["original_problem_id"] for r in multilingual if r["question_language"] == "en"}:
        raise ValueError("Exclusions or retained IDs differ from approvals")
    if Counter(r["question_language"] for r in multilingual) != {
        "en": 50,
        "zh": 50,
        "de": 48,
        "ja": 49,
    }:
        raise ValueError("Final language composition differs")
    val_ids = {
        r["original_problem_id"]
        for r in jsonl_rows(
            PHASE16G.parent / "phase16c-sft-candidate-v1/sft_export/validation.jsonl"
        )
    }
    eval_ids = {
        r["original_problem_id"]
        for r in jsonl_rows(
            ROOT / "outputs/phase14i-lite-provisional-dataset-v1/selected_questions.jsonl"
        )
    }
    official_test_ids = {
        r["original_problem_id"]
        for r in jsonl_rows(ROOT / "data/processed/gsm8k-200/splits/test.jsonl")
    }
    if set(eids) & (val_ids | eval_ids | official_test_ids):
        raise ValueError("Training/validation/evaluation leakage")
    if len(eval_ids) != 98:
        raise ValueError("Held-out evaluation set count changed")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        revision=REVISION,
        local_files_only=True,
        trust_remote_code=False,
    )
    english_by_id = {r["original_problem_id"]: r for r in english}
    approved_by_key = {
        (r["canonical_problem_id"], r["target_language"]): r
        for r in approval_rows
        if r["approval_status"] == "approved"
    }
    final_english = []
    final_multi = []
    for condition, output in ((english, final_english), (multilingual, final_multi)):
        for original_row in condition:
            row = json.loads(json.dumps(original_row))
            identifier = row["original_problem_id"]
            source_row = source_by_id[identifier]
            if (
                row["original_question"] != source_row["original_question"]
                or row["original_rationale"] != source_row["rationale"]
                or row["original_answer"] != source_row["original_answer"]
                or row["gold_answer"] != source_row["gold_answer"]
            ):
                raise ValueError(f"Source text changed in SFT row: {identifier}")
            if (
                row["original_solution_sha256"]
                != hashlib.sha256(source_row["original_answer"].encode()).hexdigest()
            ):
                raise ValueError(f"Solution hash changed: {identifier}")
            if row["target_text"] != english_by_id[identifier]["target_text"]:
                raise ValueError(f"Target text differs by condition: {identifier}")
            if row["question_language"] != "en" and condition is multilingual:
                key = (identifier, row["question_language"])
                approval = approved_by_key.get(key)
                if approval is None or row["question_text"] != approval["current_translation"]:
                    raise ValueError(f"Unapproved or changed translation: {key}")
                row["question_provenance"] = {
                    **row["question_provenance"],
                    "qa_status": "human_approved",
                    "approval_status": "approved",
                    "approval_source": "user_authorization",
                    "review_method": "AI_assisted_semantic_review",
                    "approval_scope": "Phase_16_experiment",
                    "approval_record_version": "phase16h-approved-sft-freeze-v1",
                }
            if tokenizer.apply_chat_template(row["messages"], tokenize=True) != row["input_ids"]:
                raise ValueError(f"Pinned-tokenizer IDs differ: {identifier}")
            if len(row["input_ids"]) != row["token_length"] or row["token_length"] > MAX_LENGTH:
                raise ValueError(f"Token length invalid: {identifier}")
            output.append(row)
    if any(
        r["question_provenance"].get("qa_status") != "human_approved"
        for r in final_multi
        if r["question_language"] != "en"
    ):
        raise ValueError("Retained translation missing approval provenance")

    prior_stats = json.loads((PHASE16G / "final_token_comparison.json").read_text())
    stats = {"english_only": summarize(final_english), "multilingual": summarize(final_multi)}
    if any(
        stats[name]["total_tokens"] != prior_stats["conditions"][name]["total_tokens"]
        for name in stats
    ):
        raise ValueError("Approval metadata unexpectedly changed token counts")

    OUT.mkdir(parents=True)
    write_jsonl(OUT / "english_only_sft_final.jsonl", final_english)
    write_jsonl(OUT / "multilingual_sft_final.jsonl", final_multi)
    write_csv(OUT / "final_approval_record.csv", approval_rows)
    exclusion_manifest = {
        "dataset_version": "phase16h-approved-sft-freeze-v1",
        "decision_source": "explicit_user_authorization_phase16h",
        "excluded": [
            {
                "canonical_problem_id": r["canonical_problem_id"],
                "reason": r["reason"],
                "source_issue": r["issue_category"],
            }
            for r in proposals
            if r["canonical_problem_id"] in excluded
        ],
        "retained_source_issue": [
            {
                "canonical_problem_id": r["canonical_problem_id"],
                "reason": r["reason"],
                "source_issue": r["issue_category"],
            }
            for r in proposals
            if r["canonical_problem_id"] not in excluded
        ],
        "symmetric_exclusions": True,
    }
    (OUT / "final_exclusion_manifest.json").write_text(
        json.dumps(exclusion_manifest, indent=2) + "\n"
    )
    file_hashes = {
        name: sha256(OUT / name)
        for name in (
            "english_only_sft_final.jsonl",
            "multilingual_sft_final.jsonl",
            "final_approval_record.csv",
            "final_exclusion_manifest.json",
        )
    }
    config_paths = [
        ROOT / "configs/phase16a_english_lora.yaml",
        ROOT / "configs/phase16a_multilingual_lora.yaml",
        ROOT / "outputs/phase14i-lite-real-baseline-v1/config.yaml",
    ]
    manifest = {
        "dataset_version": "phase16h-approved-sft-freeze-v1",
        "training_ready": True,
        "model": MODEL,
        "model_revision": REVISION,
        "tokenizer_checkpoint": MODEL,
        "tokenizer_revision": REVISION,
        "seed": SEED,
        "max_sequence_length": MAX_LENGTH,
        "approval_provenance": {
            "source": "explicit_user_authorization",
            "method": "AI_assisted_semantic_review",
            "scope": "Phase_16_experiment",
            "independent_expert_verification": False,
            "approved_translations": 147,
            "excluded_translations": 3,
        },
        "canonical_ids": sorted(eids),
        "exclusion_ids": sorted(excluded),
        "language_counts": dict(Counter(r["question_language"] for r in final_multi)),
        "token_statistics": stats,
        "dataset_files_sha256": file_hashes,
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                PHASE16G / "reviewed_translations_candidate.csv",
                PHASE16G / "proposed_exclusions.csv",
                PHASE16G / "english_only_sft_candidate.jsonl",
                PHASE16G / "multilingual_sft_candidate.jsonl",
            )
        },
        "configuration_references": {
            str(path.relative_to(ROOT)): sha256(path) for path in config_paths
        },
        "git_commit": get_commit(),
        "git_commit_note": "No HEAD commit exists" if get_commit() is None else None,
        "validation": {
            "matched_ids": True,
            "no_duplicate_ids": True,
            "source_text_and_hashes_unchanged": True,
            "worked_solutions_and_gold_unchanged": True,
            "no_train_validation_evaluation_overlap": True,
            "approved_translations_only": True,
            "pinned_tokenization_consistent": True,
            "no_sequence_over_limit": True,
        },
    }
    (OUT / "dataset_freeze_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    report = (
        "# Phase 16H final SFT dataset validation\n\n"
        "The user explicitly approved this Phase 16 experiment's 147 retained "
        "translations after AI-assisted semantic review. This is user authorization, "
        "not independent expert linguistic verification. Three translated questions "
        "were excluded with their source problems; the fabric dimensions question "
        "was retained with its wording issue documented. The revised German park "
        "translation and original Chinese cupcakes translation are used.\n\n"
        "Both frozen conditions contain 197 identical, unique GSM8K training IDs. "
        "English-only has 197 English questions. Multilingual has 50 English, "
        "50 Chinese, 48 German, and 49 Japanese questions. Original English worked "
        "solutions and gold answers are unchanged. No training ID overlaps validation, "
        "the 98-question evaluation set, or the official test split.\n\n"
        f"Pinned Qwen token lengths: {stats}. Phase 16G token totals were "
        f"{prior_stats['conditions']['english_only']['total_tokens']} English and "
        f"{prior_stats['conditions']['multilingual']['total_tokens']} multilingual; "
        "approval metadata did not change them. Maximum length is 2048; no truncation "
        "is needed. Dataset hashes, canonical IDs, configuration references, and "
        "approval provenance are in the freeze manifest.\n\n"
        "The dataset is training-ready for this approved scope. Technical next steps: "
        "point independent CUDA training configs at these frozen files, validate the "
        "trainer environment and available GPU memory, and run a 20-step smoke test "
        "before full training. No LoRA training was started here.\n"
    )
    (OUT / "phase16h_final_validation_report.md").write_text(report)
    for path in OUT.iterdir():
        if path.is_file():
            path.chmod(0o444)
    print(
        json.dumps(
            {
                "approved": 147,
                "excluded": 3,
                "stats": stats,
                "hashes": file_hashes,
                "training_ready": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
