"""Prepare unapproved, matched Phase 16G SFT candidates and decision records."""

# ruff: noqa: E501

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

from transformers import AutoTokenizer

from multilingual_llm_reasoning.sft_tokenization import tokenize_sft

ROOT = Path(__file__).resolve().parents[1]
PHASE16F = ROOT / "outputs/phase16f-review-reconciliation-v1"
PHASE16C = ROOT / "outputs/phase16c-sft-candidate-v1/sft_export"
OUT = ROOT / "outputs/phase16g-final-dataset-candidate-v1"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"

SOURCE_DECISIONS = {
    "7e5a38bd20f5": (
        "propose_exclude",
        "source_inconsistency",
        "Tim=79 and Tim=2×John−5 imply John=42 and James=30, while gold says 25.",
    ),
    "21a067981a33": (
        "propose_retain",
        "source_terminology",
        "Two pieces called squares have unequal side lengths; dimensions and area arithmetic remain interpretable.",
    ),
    "319132f97087": (
        "propose_exclude",
        "source_inconsistency",
        "Question asks cumulative savings; worked solution reports 60 as a weekly amount.",
    ),
    "92eae8e3cce9": (
        "propose_exclude",
        "source_denominator_ambiguity",
        "Question compares goals against Barca with all league goals; gold uses 20% of 300 against Barca.",
    ),
}

GOLD_SENSITIVITY = {
    "b3bc648b8e7d": "Seven refills versus seven total bottlefuls changes the remaining water (4 versus gold 16 ounces).",
    "a02760664932": "Eight pieces total versus eight per pan changes the brownie price (4 versus gold 2 dollars).",
    "81e6554669bd": "'Times more' versus 'times as much' changes the multiplicative relationships.",
    "76b1d076a87a": "Double versus triple initial money changes the remaining balance (gold 2700 dollars).",
    "38b97a96c0db": "One-way versus round-trip travel changes gas cost (21 versus gold 42 dollars).",
    "6590578b65aa": "A month is treated as exactly four weeks in the gold solution.",
    "a1c6e833622c": "Per-person park admission and grandmother pricing are assumed in the gold solution, not explicit in the question.",
    "06dfac8eae13": "The transfer to Jen is omitted; the final chain implies Jen's count, but the source story is incomplete.",
}


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def jsonl_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def save_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def xlsx_rows(path: Path) -> list[dict[str, str]]:
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Revised workbook ZIP is invalid")
        strings_root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
        strings = [
            "".join(t.text or "" for t in item.findall(".//x:t", namespace))
            for item in strings_root.findall("x:si", namespace)
        ]
        sheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        values = []
        for row in sheet.findall(".//x:sheetData/x:row", namespace):
            cells = {}
            for cell in row.findall("x:c", namespace):
                reference = cell.attrib["r"]
                letters = "".join(char for char in reference if char.isalpha())
                column = 0
                for letter in letters:
                    column = column * 26 + ord(letter) - ord("A") + 1
                value = cell.find("x:v", namespace)
                if value is None:
                    text = ""
                elif cell.attrib.get("t") == "s":
                    text = strings[int(value.text)]
                else:
                    text = value.text or ""
                cells[column - 1] = text
            values.append(cells)
    headers = [values[0].get(index, "") for index in range(max(values[0]) + 1)]
    return [
        {header: row.get(index, "") for index, header in enumerate(headers)} for row in values[1:]
    ]


def run() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    original = csv_rows(PHASE16F / "phase16f_consolidated_review.csv")
    changes = csv_rows(PHASE16F / "phase16f_translation_change_log.csv")
    by_key = {(r["canonical_problem_id"], r["target_language"]): r for r in original}
    if len(original) != 150 or len(by_key) != 150 or len(changes) != 2:
        raise ValueError("Review input counts or keys are invalid")
    workbook_path = PHASE16F / "phase16f_review_workbook_revised.xlsx"
    revised_rows = xlsx_rows(workbook_path)
    revised_by_key = {(r["canonical_problem_id"], r["target_language"]): r for r in revised_rows}
    if len(revised_by_key) != 150 or set(revised_by_key) != set(by_key):
        raise ValueError("Revised workbook rows differ from original review keys")
    change_by_key = {(r["canonical_problem_id"], r["target_language"]): r for r in changes}
    if len(change_by_key) != 2:
        raise ValueError("Duplicate change-log key")
    for key, row in change_by_key.items():
        if key not in by_key or row["previous_translation"] != by_key[key]["current_translation"]:
            raise ValueError(f"Change log does not match original translation: {key}")
        if row["approval_status"] != "pending":
            raise ValueError("Change log claims approval")
    german = next(k for k, r in change_by_key.items() if r["action"] == "revised")
    chinese = next(k for k, r in change_by_key.items() if r["action"] == "retained")
    if german[1] != "de" or chinese[1] != "zh":
        raise ValueError("Unexpected language in change log")
    if change_by_key[chinese]["updated_translation"] != by_key[chinese]["current_translation"]:
        raise ValueError("Retained Chinese translation changed")
    for key, row in revised_by_key.items():
        expected = change_by_key.get(key, {}).get(
            "updated_translation", by_key[key]["current_translation"]
        )
        if row["current_translation"] != expected:
            raise ValueError(f"Revised workbook translation mismatch: {key}")
        for field in (
            "english_original",
            "english_worked_solution",
            "gold_answer",
            "source_sha256",
        ):
            if row[field] != by_key[key][field]:
                raise ValueError(f"Source field changed: {key}/{field}")
        if hashlib.sha256(row["english_original"].encode()).hexdigest() != row["source_sha256"]:
            raise ValueError(f"Canonical source question hash mismatch: {key}")
        if row["final_approval_status"] != "pending" or row["reviewer_decision"]:
            raise ValueError(f"Unexpected approval: {key}")

    source = jsonl_rows(ROOT / "data/processed/gsm8k-200/subsets/train.jsonl")
    source_by_id = {r["original_problem_id"]: r for r in source}
    suffix_ids = {
        suffix: [identifier for identifier in source_by_id if identifier.endswith(suffix)]
        for suffix in SOURCE_DECISIONS
    }
    if any(len(ids) != 1 for ids in suffix_ids.values()):
        raise ValueError("Source issue canonical ID lookup failed")
    decisions = {ids[0]: SOURCE_DECISIONS[suffix] for suffix, ids in suffix_ids.items()}
    excluded = {
        identifier
        for identifier, (action, _, _) in decisions.items()
        if action == "propose_exclude"
    }
    if len(excluded) != 3:
        raise ValueError("Expected three proposed exclusions")
    exclusion_rows = []
    for identifier, (action, category, reason) in decisions.items():
        item = source_by_id[identifier]
        exclusion_rows.append(
            {
                "canonical_problem_id": identifier,
                "proposed_action": action,
                "issue_category": category,
                "reason": reason,
                "english_original": item["original_question"],
                "english_worked_solution": item["rationale"],
                "gold_answer": item["gold_answer"],
                "human_decision": "",
                "approval_status": "pending",
            }
        )
    exclusion_rows.sort(key=lambda r: r["canonical_problem_id"])

    reviewed = []
    decision_rows = []
    for row in original:
        key = (row["canonical_problem_id"], row["target_language"])
        updated = revised_by_key[key]["current_translation"]
        action = change_by_key.get(key, {}).get("action", "unchanged")
        review = {
            **row,
            "current_translation": updated,
            "translation_action": action,
            "approval_status": "pending",
        }
        reviewed.append(review)
        decision = decisions.get(key[0])
        decision_rows.append(
            {
                "canonical_problem_id": key[0],
                "language": key[1],
                "english_original": row["english_original"],
                "current_translation": updated,
                "translation_action": action,
                "phase16f_assessment": row["phase16f_assessment"],
                "original_qa_flags": row["original_qa_flags"],
                "phase16e_review_notes": row["phase16e_review_notes"],
                "source_decision_proposal": decision[0] if decision else "",
                "source_decision_reason": decision[2] if decision else "",
                "reviewer_decision": "",
                "reviewer_comments": "",
                "translation_approval_status": "pending",
                "source_decision_status": "pending" if decision else "not_applicable",
            }
        )

    english = jsonl_rows(PHASE16C / "english_train.jsonl")
    multilingual = jsonl_rows(PHASE16C / "multilingual_train.jsonl")
    if len(english) != 200 or len(multilingual) != 200:
        raise ValueError("Original SFT count changed")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        revision=REVISION,
        local_files_only=True,
        trust_remote_code=False,
    )
    english_candidate = [r.copy() for r in english if r["original_problem_id"] not in excluded]
    multi_candidate = [r.copy() for r in multilingual if r["original_problem_id"] not in excluded]
    if len(english_candidate) != len(multi_candidate) or len(english_candidate) != 197:
        raise ValueError("Symmetric exclusion failed")
    for row in multi_candidate:
        if (row["original_problem_id"], row["question_language"]) == german:
            updated = change_by_key[german]["updated_translation"]
            row["question_text"] = updated
            row["messages"][1]["content"] = updated
            row.update(tokenize_sft(tokenizer, row["messages"]))
            row["length_status"] = "ok" if row["token_length"] <= 2048 else "overlength"
            row["question_provenance"] = {
                **row["question_provenance"],
                "qa_status": "pending_human_approval",
                "translation_action": "phase16g_documented_revision",
            }
    eids = [r["original_problem_id"] for r in english_candidate]
    mids = [r["original_problem_id"] for r in multi_candidate]
    if len(set(eids)) != 197 or len(set(mids)) != 197 or set(eids) != set(mids):
        raise ValueError("Candidate IDs do not align")
    if set(eids) & excluded:
        raise ValueError("Proposed exclusion leaked into candidate")
    test_ids = {
        r["original_problem_id"]
        for r in jsonl_rows(
            ROOT / "outputs/phase14i-lite-provisional-dataset-v1/selected_questions.jsonl"
        )
    }
    validation_ids = {r["original_problem_id"] for r in jsonl_rows(PHASE16C / "validation.jsonl")}
    if set(eids) & (test_ids | validation_ids):
        raise ValueError("Training/evaluation leakage")
    for condition in (english_candidate, multi_candidate):
        for row in condition:
            original_row = source_by_id[row["original_problem_id"]]
            if (
                row["original_question"] != original_row["original_question"]
                or row["original_rationale"] != original_row["rationale"]
                or row["original_answer"] != original_row["original_answer"]
            ):
                raise ValueError("Original source text changed")
            if (
                row["original_solution_sha256"]
                != hashlib.sha256(original_row["original_answer"].encode()).hexdigest()
            ):
                raise ValueError("Original solution hash changed")
            if tokenizer.apply_chat_template(row["messages"], tokenize=True) != row["input_ids"]:
                raise ValueError("Tokenization mismatch")
            if row["token_length"] > 2048:
                raise ValueError("Candidate exceeds sequence limit")
    english_by_id = {r["original_problem_id"]: r for r in english_candidate}
    for row in multi_candidate:
        peer = english_by_id[row["original_problem_id"]]
        if (
            row["target_text"] != peer["target_text"]
            or row["original_rationale"] != peer["original_rationale"]
        ):
            raise ValueError("Matched targets changed")

    language_counts = dict(Counter(r["question_language"] for r in multi_candidate))
    totals = {
        name: {
            "count": len(rows),
            "total_tokens": sum(r["token_length"] for r in rows),
            "mean_tokens": sum(r["token_length"] for r in rows) / len(rows),
            "max_tokens": max(r["token_length"] for r in rows),
            "over_2048": sum(r["token_length"] > 2048 for r in rows),
        }
        for name, rows in (("english_only", english_candidate), ("multilingual", multi_candidate))
    }
    stats = {
        "tokenizer_checkpoint": "Qwen/Qwen2.5-3B-Instruct",
        "tokenizer_revision": REVISION,
        "max_sequence_length": 2048,
        "candidate_ids": 197,
        "proposed_exclusions": 3,
        "multilingual_language_counts": language_counts,
        "conditions": totals,
        "multilingual_minus_english_tokens": totals["multilingual"]["total_tokens"]
        - totals["english_only"]["total_tokens"],
        "training_ready": False,
        "explicitly_approved_translations": 0,
    }
    report = (
        "# Phase 16G candidate validation\n\n"
        "The revised workbook and change log agree with Phase 16F. One documented German "
        "park-admission translation was applied to this new candidate; the Chinese "
        "cupcakes translation was retained unchanged. Neither is human-approved. "
        "All English originals, worked solutions, gold answers, and source hashes remain unchanged.\n\n"
        "Three source questions are proposed for symmetric exclusion: the inconsistent "
        "James/John/Tim ages, Jay's noncumulative savings solution, and the Barca "
        "denominator mismatch. The fabric dimensions question is proposed for retention "
        "with a documented terminology issue. These are pending decisions. Additional "
        "source ambiguities, including refill count, round-trip travel, and 'times more' "
        "wording, may affect gold-answer interpretation and need adjudication.\n\n"
        "## Retained-source gold-answer risks\n\n"
        + "".join(
            f"- `{next(identifier for identifier in source_by_id if identifier.endswith(suffix))}`: {reason}\n"
            for suffix, reason in GOLD_SENSITIVITY.items()
        )
        + "\nThese risks are not additional exclusions. Each requires a source decision "
        "before claiming a fully reliable gold answer.\n\n"
        f"Both proposed SFT conditions have 197 identical, unique canonical IDs. "
        f"Multilingual question-language counts: {language_counts}. No replacement was "
        "introduced, and train/validation/evaluation IDs remain disjoint. The same "
        "English worked solutions and targets are retained in both conditions.\n\n"
        f"Pinned Qwen token comparison: {totals}. Multilingual minus English total tokens: "
        f"{stats['multilingual_minus_english_tokens']}. No candidate exceeds 2048 tokens "
        "or uses truncation.\n\n"
        "All 150 translation approvals remain pending; three exclusions and the proposed "
        "fabric retention also require source decisions. These files are candidates only. "
        "No unapproved translation is labeled training-ready. CUDA LoRA training must wait "
        "for explicit decisions and a fresh validated final export.\n"
    )
    OUT.mkdir(parents=True)
    save_csv(OUT / "reviewed_translations_candidate.csv", reviewed)
    save_csv(OUT / "proposed_exclusions.csv", exclusion_rows)
    save_csv(OUT / "approval_decisions_required.csv", decision_rows)
    save_jsonl(OUT / "english_only_sft_candidate.jsonl", english_candidate)
    save_jsonl(OUT / "multilingual_sft_candidate.jsonl", multi_candidate)
    (OUT / "final_token_comparison.json").write_text(json.dumps(stats, indent=2) + "\n")
    (OUT / "phase16g_validation_report.md").write_text(report)
    translation_manifest = {
        "dataset_version": "phase16g-reviewed-translations-candidate-v1",
        "source_review_sha256": hashlib.sha256(
            (PHASE16F / "phase16f_consolidated_review.csv").read_bytes()
        ).hexdigest(),
        "revised_workbook_sha256": hashlib.sha256(workbook_path.read_bytes()).hexdigest(),
        "change_log_sha256": hashlib.sha256(
            (PHASE16F / "phase16f_translation_change_log.csv").read_bytes()
        ).hexdigest(),
        "canonical_training_ids_original": 200,
        "translation_records": len(reviewed),
        "documented_translation_revisions_applied": 1,
        "explicit_translation_approvals": 0,
        "proposed_excluded_ids": sorted(excluded),
        "candidate_training_ids": 197,
        "training_ready": False,
    }
    (OUT / "translation_manifest.json").write_text(
        json.dumps(translation_manifest, indent=2) + "\n"
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    run()
