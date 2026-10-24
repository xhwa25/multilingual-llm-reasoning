"""Export and validate the fixed 150-question training translation handoff."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SFT = ROOT / "data/sft/gsm8k-200"
EVAL = ROOT / "outputs/phase14i-lite-provisional-dataset-v1/manifest.json"
OUT = ROOT / "outputs/phase16b-training-translations-v1"
LANGS = ("zh", "de", "ja")
FIELDS = [
    "canonical_problem_id",
    "target_language",
    "english_original",
    "english_worked_solution",
    "gold_answer",
    "source_sha256",
    "translated_question",
    "numeric_qa_note",
    "mathematical_equivalence",
    "reviewer_comments",
    "approval_status",
]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def source_rows() -> list[dict[str, str]]:
    manifest = json.loads((SFT / "manifest.json").read_text())
    if manifest["status"] != "partial" or len(manifest["selected_train_ids"]) != 200:
        raise ValueError("Expected original 200-ID partial SFT manifest")
    evaluations = set(json.loads(EVAL.read_text())["selection"]["problem_ids"])
    validations = set(manifest["selected_validation_ids"])
    selected = set(manifest["selected_train_ids"])
    if selected & (evaluations | validations):
        raise ValueError("Training/evaluation or training/validation leakage")
    examples = {
        r["original_problem_id"]: r
        for r in map(json.loads, (SFT / "english_train.jsonl").read_text().splitlines())
    }
    if set(examples) != selected:
        raise ValueError("English SFT IDs differ from selected IDs")
    rows = []
    for identity in manifest["selected_train_ids"]:
        lang = manifest["language_assignments"][identity]
        if lang == "en":
            continue
        if lang not in LANGS:
            raise ValueError("Unexpected target language")
        example = examples[identity]
        if example["split"] != "train" or example["source_split"] != "train":
            raise ValueError("Non-training example in SFT export")
        source = example["original_question"]
        rows.append(
            {
                "canonical_problem_id": identity,
                "target_language": lang,
                "english_original": source,
                "english_worked_solution": example["original_answer"],
                "gold_answer": example["gold_answer"],
                "source_sha256": digest(source),
                "translated_question": "",
                "numeric_qa_note": "",
                "mathematical_equivalence": "pending",
                "reviewer_comments": "",
                "approval_status": "pending",
            }
        )
    if len(rows) != 150 or Counter(r["target_language"] for r in rows) != {x: 50 for x in LANGS}:
        raise ValueError("Assigned translation composition differs from 50/50/50")
    return rows


def export() -> None:
    if OUT.exists():
        raise ValueError("Refusing to overwrite versioned handoff")
    OUT.mkdir(parents=True)
    rows = source_rows()
    for name in ("translation_worksheet.csv", "translation_import_template.csv"):
        with (OUT / name).open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    with (OUT / "translation_import_template.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1,
        "version": "phase16b-training-translations-v1",
        "status": "pending_translations",
        "canonical_ids": 150,
        "language_counts": dict(Counter(r["target_language"] for r in rows)),
        "source_sft_manifest_sha256": hashlib.sha256(
            (SFT / "manifest.json").read_bytes()
        ).hexdigest(),
        "evaluation_manifest_sha256": hashlib.sha256(EVAL.read_bytes()).hexdigest(),
        "approval_status": "pending",
        "training_ready": False,
        "files": {
            name: {
                "bytes": (OUT / name).stat().st_size,
                "sha256": hashlib.sha256((OUT / name).read_bytes()).hexdigest(),
            }
            for name in (
                "translation_worksheet.csv",
                "translation_import_template.csv",
                "translation_import_template.jsonl",
            )
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def validate(path: Path) -> dict:
    expected = {(r["canonical_problem_id"], r["target_language"]): r for r in source_rows()}
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if set(reader.fieldnames or []) != set(FIELDS):
            raise ValueError("Import CSV columns differ from template")
        imported = list(reader)
    seen = set()
    issues = []
    for row in imported:
        pair = (row["canonical_problem_id"], row["target_language"])
        if pair in seen:
            issues.append({"pair": pair, "issue": "duplicate_id_language"})
        seen.add(pair)
        source = expected.get(pair)
        if source is None:
            issues.append({"pair": pair, "issue": "unexpected_or_leaked_id"})
            continue
        for key in ("english_original", "english_worked_solution", "gold_answer", "source_sha256"):
            if row[key] != source[key]:
                issues.append({"pair": pair, "issue": f"changed_{key}"})
        translated = row["translated_question"].strip()
        if not translated:
            issues.append({"pair": pair, "issue": "missing_translation"})
        source_nums = sorted(re.findall(r"(?<![0-9])\d+(?:[.,]\d+)?", source["english_original"]))
        target_nums = sorted(re.findall(r"(?<![0-9])\d+(?:[.,]\d+)?", translated))
        if translated and source_nums != target_nums and not row["numeric_qa_note"].strip():
            issues.append({"pair": pair, "issue": "numeric_tokens_differ_review_required"})
        if row["approval_status"] == "approved":
            if row["mathematical_equivalence"] != "confirmed" or not translated:
                issues.append({"pair": pair, "issue": "approval_without_confirmed_equivalence"})
            if source_nums != target_nums and not row["numeric_qa_note"].strip():
                issues.append({"pair": pair, "issue": "approved_numeric_mismatch_without_note"})
        elif row["approval_status"] != "pending":
            issues.append({"pair": pair, "issue": "invalid_approval_status"})
    for pair in expected.keys() - seen:
        issues.append({"pair": pair, "issue": "missing_id_language"})
    report = {
        "rows": len(imported),
        "expected_rows": 150,
        "ready": not issues and all(r["approval_status"] == "approved" for r in imported),
        "issues": issues,
    }
    return report


def publish(path: Path, output_dir: Path) -> None:
    report = validate(path)
    if not report["ready"]:
        raise ValueError("Import is incomplete or unapproved; inspect validation report")
    if output_dir.exists():
        raise ValueError("Refusing to overwrite translation artifact")
    expected = {(r["canonical_problem_id"], r["target_language"]): r for r in source_rows()}
    with path.open(encoding="utf-8-sig", newline="") as file:
        imported = list(csv.DictReader(file))
    variants = []
    for row in imported:
        source = expected[(row["canonical_problem_id"], row["target_language"])]
        variants.append(
            {
                "original_problem_id": row["canonical_problem_id"],
                "language": row["target_language"],
                "split": "train",
                "source_split": "train",
                "source_text": source["english_original"],
                "translated_text": row["translated_question"],
                "gold_answer": source["gold_answer"],
                "qa_status": "human_approved",
                "approval_status": "approved",
                "translation_provider": "reviewed_import",
                "translation_model": None,
                "translation_prompt_version": "external_draft_unknown",
                "reviewer_comments": row["reviewer_comments"],
                "numeric_qa_note": row["numeric_qa_note"],
            }
        )
    output_dir.mkdir(parents=True)
    payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in variants)
    (output_dir / "variants.jsonl").write_text(payload)
    prep = ROOT / "data/processed/gsm8k-200/manifest.json"
    manifest = {
        "schema_version": 1,
        "complete": True,
        "mock": False,
        "input_manifest_sha256": hashlib.sha256(prep.read_bytes()).hexdigest(),
        "review_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "artifacts": {
            "variants": {"path": "variants.jsonl", "count": 150, "sha256": digest(payload)}
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("export", "validate", "publish"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.command == "export":
        export()
    elif args.command == "validate":
        if args.input is None:
            parser.error("validate requires --input")
        print(json.dumps(validate(args.input), indent=2, ensure_ascii=False))
    else:
        if args.input is None or args.output_dir is None:
            parser.error("publish requires --input and --output-dir")
        publish(args.input, args.output_dir)
