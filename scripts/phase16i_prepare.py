"""Build a trainer manifest referencing, never copying, Phase 16H frozen SFT files."""

# ruff: noqa: E501

import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "outputs/phase16h-approved-sft-freeze-v1"
PREVIOUS = ROOT / "outputs/phase16c-sft-candidate-v1/sft_export"
OUT = ROOT / "outputs/phase16i-training-preparation-v1"
EXPECTED_HASHES = {
    "english_only_sft_final.jsonl": "6e59b5344db55eb84f87c9dc9059587fbd00dae0edcea967a76d37705b8ed881",
    "multilingual_sft_final.jsonl": "9b5c813da240229a31a7e57d1bc8a8e4cb9e8a410a837c38b01db23202df2021",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    freeze_path = FREEZE / "dataset_freeze_manifest.json"
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("training_ready") is not True:
        raise ValueError("Phase 16H freeze is not training-ready")
    for name, expected in EXPECTED_HASHES.items():
        if digest(FREEZE / name) != expected or freeze["dataset_files_sha256"][name] != expected:
            raise ValueError(f"Frozen dataset hash mismatch: {name}")
    english = rows(FREEZE / "english_only_sft_final.jsonl")
    multilingual = rows(FREEZE / "multilingual_sft_final.jsonl")
    validation_path = PREVIOUS / "validation.jsonl"
    validation = rows(validation_path)
    if len(english) != 197 or len(multilingual) != 197 or len(validation) != 100:
        raise ValueError("Frozen training or validation counts changed")
    old_manifest = json.loads((PREVIOUS / "manifest.json").read_text())
    if digest(validation_path) != old_manifest["artifacts"]["validation"]["sha256"]:
        raise ValueError("Validation set hash changed")
    train_ids = sorted(r["original_problem_id"] for r in english)
    multi_ids = sorted(r["original_problem_id"] for r in multilingual)
    if train_ids != multi_ids or train_ids != freeze["canonical_ids"]:
        raise ValueError("Frozen canonical IDs differ")
    language_counts = dict(Counter(r["question_language"] for r in multilingual))
    if language_counts != {"en": 50, "zh": 50, "de": 48, "ja": 49}:
        raise ValueError("Approved language distribution differs")
    artifacts = {
        "english_train": {
            "path": "../phase16h-approved-sft-freeze-v1/english_only_sft_final.jsonl",
            "sha256": EXPECTED_HASHES["english_only_sft_final.jsonl"],
            "count": 197,
        },
        "multilingual_train": {
            "path": "../phase16h-approved-sft-freeze-v1/multilingual_sft_final.jsonl",
            "sha256": EXPECTED_HASHES["multilingual_sft_final.jsonl"],
            "count": 197,
        },
        "validation": {
            "path": "../phase16c-sft-candidate-v1/sft_export/validation.jsonl",
            "sha256": digest(validation_path),
            "count": 100,
        },
    }

    def summary(data: list[dict]) -> dict:
        return {
            "count": len(data),
            "question_language_counts": dict(Counter(row["question_language"] for row in data)),
            "prompt_tokens": sum(row["prompt_tokens"] for row in data),
            "supervised_response_tokens": sum(row["target_tokens"] for row in data),
            "total_tokens": sum(row["token_length"] for row in data),
        }

    manifest = {
        "schema_version": 2,
        "status": "ready",
        "training_ready": True,
        "approved_freeze_manifest": "../phase16h-approved-sft-freeze-v1/dataset_freeze_manifest.json",
        "approved_freeze_sha256": digest(freeze_path),
        "approved_language_counts": language_counts,
        "artifacts": artifacts,
        "tokenizer": old_manifest["tokenizer"],
        "provenance": {
            **old_manifest["provenance"],
            "allowed_train_ids": train_ids,
            "mock_questions": False,
            "approval_source": "user_authorization",
            "review_method": "AI_assisted_semantic_review",
        },
        "diagnostics": {
            "english_train": summary(english),
            "multilingual_train": summary(multilingual),
            "validation": summary(validation),
        },
        "selected_train_ids": train_ids,
        "selected_validation_ids": sorted(row["original_problem_id"] for row in validation),
    }
    if manifest["tokenizer"]["resolved_revision"] != freeze["tokenizer_revision"]:
        raise ValueError("Pinned tokenizer revision differs")
    OUT.mkdir(parents=True)
    (OUT / "sft_training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    baseline = yaml.safe_load((ROOT / "configs/phase16a_english_lora.yaml").read_text())
    for language, condition in (
        ("english", "english_train"),
        ("multilingual", "multilingual_train"),
    ):
        for smoke in (False, True):
            config = dict(baseline)
            label = f"phase16i_{language}_{'smoke' if smoke else 'lora'}"
            config["name"] = label
            config["condition"] = condition
            config["sft_manifest"] = (
                "../outputs/phase16i-training-preparation-v1/sft_training_manifest.json"
            )
            config["output_dir"] = f"../outputs/training/{label}"
            path = ROOT / "configs" / f"{label}.yaml"
            if path.exists():
                raise FileExistsError(path)
            path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(OUT / "sft_training_manifest.json")


if __name__ == "__main__":
    main()
