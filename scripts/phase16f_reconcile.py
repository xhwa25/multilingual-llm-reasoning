"""Reconcile Phase 16E recommendations without converting them to approvals."""

# ruff: noqa: E501

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/phase16c-sft-candidate-v1"
REVIEW = ROOT / "outputs/phase16d-translation-review-v1/review_queue.csv"
OUTPUT = ROOT / "outputs/phase16f-review-reconciliation-v1"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"

# Keys are stable canonical-hash suffixes, checked for uniqueness before use.
# These classifications are evidence-based recommendations, never approvals.
DISPUTES = {
    "7a99be16b80c": (
        "source_ambiguity",
        "'half times younger' is ambiguous; 40/2=20 matches the gold 16.",
    ),
    "7e5a38bd20f5": (
        "source_inconsistency",
        "79=2×John−5 gives John=42 and James=30, not gold 25; the worked solution reverses the five-year relation.",
    ),
    "c5361a66a668": (
        "source_ambiguity",
        "Rainfall depth is treated as tank fill depth; collection geometry is unstated in the source.",
    ),
    "b3bc648b8e7d": (
        "source_ambiguity",
        "Seven refills may follow an initial bottle, but the gold solution counts seven bottlefuls only.",
    ),
    "81e7f6048980": (
        "translation_correct",
        "German 'Junge' conveys rabbit offspring; the source's 'kittens' wording is awkward.",
    ),
    "a02760664932": (
        "source_ambiguity",
        "Source does not explicitly say eight pieces per pan; the worked solution uses 2×8.",
    ),
    "81e6554669bd": (
        "source_ambiguity",
        "Source 'times more' is ambiguous; German multiplicative reading matches the worked solution.",
    ),
    "a1c6e833622c": (
        "translation_needs_revision",
        "German adds 'per person' and applies the adult price to grandmother; both are in the solution but unstated in the question.",
    ),
    "ad96a932ce13": (
        "translation_correct",
        "Chinese preserves the unusual per-table 2/5 relation, matching 500×(2/5×500).",
    ),
    "76b1d076a87a": (
        "source_ambiguity",
        "'Two times more' can mean double or triple; German chooses double as in gold 2700.",
    ),
    "1c095f41209f": (
        "translation_needs_revision",
        "Chinese adds an explicit 20-person total absent from the source, though 19 friends plus daughter implies it.",
    ),
    "3cb265333ffc": (
        "source_ambiguity",
        "Paid lessons after the free first lesson require an ordering convention; solution charges lessons 13, 15, 17, 19.",
    ),
    "742e147f099d": (
        "translation_correct",
        "Japanese leaf-per-cent and 200 clover stems preserve the source; 150×3+48×2+2×4=554.",
    ),
    "400cd8432681": (
        "source_ambiguity",
        "'Half of them' refers to mixed supplies; solution counts 24 paper pieces plus six glue bottles.",
    ),
    "21a067981a33": (
        "source_inconsistency",
        "Source calls 8×5 and 10×7 pieces squares; German calls them rectangles, preserving dimensions and area.",
    ),
    "38b97a96c0db": (
        "source_ambiguity",
        "Source says one-way distance but asks trip cost; gold assumes a round trip, not stated explicitly.",
    ),
    "401d5c5737f8": (
        "source_ambiguity",
        "Penalty timing at completion on day six requires an interval convention; translation preserves source.",
    ),
    "6590578b65aa": (
        "source_ambiguity",
        "One month at 30 hours/week assumes four weeks; translation preserves this source assumption.",
    ),
    "bb8938fa75b0": (
        "translation_correct",
        "Chinese preserves both half-year weekly rates; solution uses 26 weeks per half.",
    ),
    "06dfac8eae13": (
        "source_ambiguity",
        "Source never states how Jen received suckers after Sienna gave Bailey half; German does not supply that link.",
    ),
    "92eae8e3cce9": (
        "source_inconsistency",
        "Source switches from all league goals to goals against Barca; gold uses 20% of 300 against Barca.",
    ),
    "023cb2fc8964": (
        "translation_correct",
        "Japanese preserves four distinct names; source rationale has Zoran typo and 'average of three' typo, but computes four.",
    ),
    "33c86ebc1733": (
        "translation_correct",
        "German preserves two 20-minute trips per day; 3000/40=75.",
    ),
    "e0a293b75a78": (
        "translation_correct",
        "German preserves additive 4 minutes per grass stain plus seven for marinara; 3×4+7=19.",
    ),
    "319132f97087": (
        "source_inconsistency",
        "Question asks total saved in a month; rationale reports a weekly amount of 60, not a cumulative total.",
    ),
}

CORRECTIONS = {
    "a1c6e833622c": (
        "Im Stadtpark gibt es verschiedene Attraktionen für Kinder und ihre Eltern. "
        "Der Eintritt in den Park kostet 5 Dollar. Für jede Attraktion muss man "
        "zusätzlich bezahlen: Eine Karte kostet für Kinder 2 Dollar und für "
        "Eltern 4 Dollar. Wie viel bezahlt eine Familie mit 4 Kindern, "
        "deren Eltern und Großmutter für den Park und eine Attraktion?",
        "Removes per-person admission and grandmother's adult price, which are not explicit in the English question; preserves all source quantities.",
    ),
    "1c095f41209f": (
        "卡拉为女儿的生日派对烤纸杯蛋糕，每批65个，一共烤了45批。随后，她从每一批中取出5个喂给狗。卡拉的女儿有19个朋友，如果这些朋友和女儿平均分掉剩下的蛋糕，那么女儿的每位朋友吃了多少个纸杯蛋糕？",
        "Removes the added explicit total of 20 people while preserving the source's 19 friends plus daughter relationship.",
    ),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(inputs: tuple[Path, Path]) -> None:
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    recommendations = [row for path in inputs for row in read_csv(path)]
    originals = read_csv(SOURCE / "imported_draft_original.csv")
    queue = read_csv(REVIEW)
    if len(recommendations) != 150 or len(originals) != 150 or len(queue) != 150:
        raise ValueError("Expected 150 source, queue, and recommendation rows")
    by_key = {(r["canonical_problem_id"], r["target_language"]): r for r in originals}
    queue_by_key = {(r["canonical_problem_id"], r["target_language"]): r for r in queue}
    keys = [(r["canonical_problem_id"], r["target_language"]) for r in recommendations]
    if len(set(keys)) != 150 or set(keys) != set(by_key) or set(keys) != set(queue_by_key):
        raise ValueError("Missing, duplicate, or unexpected review key")
    if Counter(k[1] for k in keys) != {"zh": 50, "de": 50, "ja": 50}:
        raise ValueError("Language counts are not 50/50/50")
    suffixes = {suffix: [k for k in keys if k[0].endswith(suffix)] for suffix in DISPUTES}
    if any(len(matches) != 1 for matches in suffixes.values()):
        raise ValueError("Disputed canonical suffix is missing or ambiguous")
    by_suffix = {matches[0]: DISPUTES[suffix] for suffix, matches in suffixes.items()}
    corrections = {suffixes[suffix][0]: value for suffix, value in CORRECTIONS.items()}
    rows = []
    proposed = []
    for rec in recommendations:
        key = (rec["canonical_problem_id"], rec["target_language"])
        source, prior = by_key[key], queue_by_key[key]
        checks = {
            "english_original": source["english_original"],
            "english_worked_solution": source["english_worked_solution"],
            "gold_answer": source["gold_answer"],
            "source_sha256": source["source_sha256"],
            "current_translation": source["translated_question"],
            "qa_flags": prior["qa_flags"],
        }
        if any(rec[field] != value for field, value in checks.items()):
            raise ValueError(f"Source or QA mismatch for {key}")
        if rec["approval_status"] != "pending" or rec["reviewer_decision"].strip():
            raise ValueError(f"Unexpected approval or decision in {key}")
        classification, evidence = by_suffix.get(
            key,
            (
                "translation_correct",
                "Phase 16E found no specific discrepancy; independent human confirmation remains required.",
            ),
        )
        correction, justification = corrections.get(key, ("", ""))
        assessment = rec.get("assistant_semantic_assessment") or rec.get(
            "semantic_review_assessment", ""
        )
        notes = rec.get("assistant_review_notes") or rec.get("semantic_review_notes", "")
        row = {
            "canonical_problem_id": key[0],
            "target_language": key[1],
            "english_original": source["english_original"],
            "current_translation": source["translated_question"],
            "proposed_correction": correction,
            "gold_answer": source["gold_answer"],
            "english_worked_solution": source["english_worked_solution"],
            "source_sha256": source["source_sha256"],
            "original_qa_flags": prior["qa_flags"],
            "phase16e_review_recommendation": assessment,
            "phase16e_review_notes": notes,
            "phase16e_review_issue_type": rec.get("review_issue_type", ""),
            "phase16e_suggested_action": rec.get("suggested_action", ""),
            "phase16f_assessment": classification,
            "issue_category": classification,
            "phase16f_evidence": evidence,
            "correction_justification": justification,
            "existing_reviewer_comments": rec["existing_reviewer_comments"],
            "phase16e_new_reviewer_comments": rec["new_reviewer_comments"],
            "reviewer_decision": "",
            "reviewer_comments": "",
            "final_approval_status": "pending",
        }
        rows.append(row)
        if correction:
            proposed.append(
                {
                    k: row[k]
                    for k in (
                        "canonical_problem_id",
                        "target_language",
                        "english_original",
                        "current_translation",
                        "proposed_correction",
                        "gold_answer",
                        "correction_justification",
                        "final_approval_status",
                    )
                }
            )
    rows.sort(key=lambda r: (r["canonical_problem_id"], r["target_language"]))

    english = read_jsonl(SOURCE / "sft_export/english_train.jsonl")
    multi = read_jsonl(SOURCE / "sft_export/multilingual_train.jsonl")
    eval_rows = read_jsonl(
        ROOT / "outputs/phase14i-lite-provisional-dataset-v1/selected_questions.jsonl"
    )
    val_rows = read_jsonl(SOURCE / "sft_export/validation.jsonl")
    eids = [r["original_problem_id"] for r in english]
    mids = [r["original_problem_id"] for r in multi]
    if len(eids) != 200 or len(set(eids)) != 200 or len(mids) != 200 or len(set(mids)) != 200:
        raise ValueError("Duplicate SFT IDs")
    if set(eids) != set(mids) or len(eids) != 200:
        raise ValueError("Conditions are not matched")
    if Counter(r["question_language"] for r in multi) != {"en": 50, "zh": 50, "de": 50, "ja": 50}:
        raise ValueError("Multilingual language counts mismatch")
    if any(
        r["original_rationale"]
        != next(e for e in english if e["original_problem_id"] == r["original_problem_id"])[
            "original_rationale"
        ]
        for r in multi
    ):
        raise ValueError("Worked solutions differ between conditions")
    eval_ids = {r["original_problem_id"] for r in eval_rows}
    val_ids = {r["original_problem_id"] for r in val_rows}
    if set(eids) & (eval_ids | val_ids) or val_ids & eval_ids:
        raise ValueError("Train/validation/evaluation leakage")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        revision=REVISION,
        local_files_only=True,
        trust_remote_code=False,
    )
    lengths = {}
    for name, condition in (("english", english), ("multilingual", multi)):
        actual = []
        for row in condition:
            ids = tokenizer.apply_chat_template(row["messages"], tokenize=True)
            if ids != row["input_ids"] or len(ids) != row["token_length"]:
                raise ValueError(
                    f"Saved tokenization mismatch: {name}/{row['original_problem_id']}"
                )
            actual.append(len(ids))
        lengths[name] = {
            "count": len(actual),
            "total": sum(actual),
            "min": min(actual),
            "max": max(actual),
            "mean": sum(actual) / len(actual),
            "over_2048": sum(x > 2048 for x in actual),
        }

    OUTPUT.mkdir(parents=True)
    write_csv(OUTPUT / "phase16f_consolidated_review.csv", rows)
    write_csv(OUTPUT / "phase16f_proposed_corrections.csv", proposed)
    counts = Counter(r["phase16f_assessment"] for r in rows)
    report = (
        "# Phase 16F review reconciliation and SFT validation\n\n"
        "The two Phase 16E files contain recommendations, not human approvals. "
        "All 150 source texts, worked solutions, answers, hashes, translations, "
        "and original QA flags match Phase 16C/D exactly. All final approvals remain pending.\n\n"
        "Phase 16F assessment counts (recommendations, not approvals):\n\n"
        f"- No identified translation issue: {counts['translation_correct']}\n"
        f"- Translation needs revision: {counts['translation_needs_revision']}\n"
        f"- Source ambiguity: {counts['source_ambiguity']}\n"
        f"- Source inconsistency: {counts['source_inconsistency']}\n"
        f"- Unresolved classification: {counts['unresolved']}\n"
        "- Explicitly approved: 0\n\n"
        "'No identified translation issue' means no specific discrepancy was found "
        "in this first-pass review; it is not a claim of human verification. "
        f"Proposed corrections: {len(proposed)}; neither is applied to the candidate.\n\n"
        "The two clear source inconsistencies needing immediate resolution are the "
        "James/John/Tim age problem (given facts imply James 30, gold 25) and Jay's "
        "savings problem (question asks cumulative savings, rationale reports 60 as "
        "a weekly amount). Other source ambiguities and solution assumptions are "
        "listed with canonical IDs and evidence in the consolidated review.\n\n"
        "## SFT integrity\n\n"
        "Both conditions have the same 200 unique training IDs. Multilingual "
        "composition is 50 each of en/zh/de/ja; worked solutions and targets are "
        "the original English text in both conditions. Training, validation, and "
        "the 98-question evaluation IDs are disjoint. Exact pinned Qwen chat-template "
        "tokenization reproduced all saved input IDs without truncation.\n\n"
        f"Token lengths: English {lengths['english']}; multilingual {lengths['multilingual']}. "
        "Configured maximum is 2048 tokens.\n\n"
        "## Training readiness\n\n"
        "Phase 16G is blocked. No translation has explicit human approval, and "
        "source issues and proposed corrections require decisions. Do not use the "
        "provisional multilingual candidate as a final training-ready export. "
        "Review the consolidated workbook, decide each row, adjudicate the source "
        "inconsistencies without silently changing GSM8K, and revalidate any approved "
        "corrected text before export.\n"
    )
    (OUTPUT / "phase16f_validation_report.md").write_text(report)
    manifest = {
        "inputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "classifications": dict(counts),
        "explicitly_approved": 0,
        "tokenizer_revision": REVISION,
        "token_lengths": lengths,
        "train_ids": 200,
        "evaluation_ids": len(eval_ids),
        "validation_ids": len(val_ids),
        "training_ready": False,
    }
    (OUTPUT / "phase16f_validation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-01", required=True, type=Path)
    parser.add_argument("--remaining-125", required=True, type=Path)
    args = parser.parse_args()
    run((args.batch_01, args.remaining_125))
