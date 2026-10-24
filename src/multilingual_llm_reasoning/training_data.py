"""Authenticate paired SFT exports and retain their exact response-only labels."""

import hashlib
import json
from typing import Any

from .sft import validate_sft_pairs
from .training_config import TrainingConfig


def load_training_data(config: TrainingConfig) -> tuple[list[dict], list[dict], dict]:
    manifest = json.loads(config.sft_manifest.read_text())
    schema = manifest.get("schema_version")
    if schema not in (1, 2):
        raise ValueError("Unsupported SFT manifest schema_version")
    frozen = None
    if schema == 2:
        freeze_path = (config.sft_manifest.parent / manifest["approved_freeze_manifest"]).resolve()
        frozen = json.loads(freeze_path.read_text())
        if (
            frozen.get("training_ready") is not True
            or frozen.get("dataset_version") != "phase16h-approved-sft-freeze-v1"
        ):
            raise ValueError("Approved Phase 16H freeze manifest is required")
        if (
            hashlib.sha256(freeze_path.read_bytes()).hexdigest()
            != manifest["approved_freeze_sha256"]
        ):
            raise ValueError("Approved freeze manifest hash mismatch")
        if manifest.get("approved_language_counts") != frozen.get(
            "language_counts"
        ) or manifest.get("selected_train_ids") != frozen.get("canonical_ids"):
            raise ValueError("Training manifest differs from approved freeze identities")
    partial = manifest.get("status") == "partial"
    if partial and config.condition != "english_train":
        raise ValueError(
            "Multilingual SFT is missing: supply reviewed translations and complete prepare-sft"
        )
    english_ready = (
        partial
        and manifest.get("conditions", {}).get("english_train", {}).get("training_ready") is True
    )
    if not english_ready and (
        manifest.get("training_ready") is not True or manifest.get("status") != "ready"
    ):
        raise ValueError(
            "SFT data is not training-ready. Review real translations and rerun "
            "prepare-sft; mock previews cannot train."
        )
    tokenizer = manifest["tokenizer"]
    if tokenizer["checkpoint"] != config.model or tokenizer["resolved_revision"] != config.revision:
        raise ValueError(
            "Model/tokenizer revision differs from audited SFT data; regenerate consistently"
        )
    groups = {}
    names = (
        ("english_train", "validation")
        if partial
        else ("english_train", "multilingual_train", "validation")
    )
    for name in names:
        artifact = manifest["artifacts"][name]
        path = (config.sft_manifest.parent / artifact["path"]).resolve()
        if schema == 1:
            if not path.is_relative_to(config.sft_manifest.parent.resolve()):
                raise ValueError("SFT artifact escapes manifest directory")
        elif name in ("english_train", "multilingual_train"):
            expected_name = (
                "english_only_sft_final.jsonl"
                if name == "english_train"
                else "multilingual_sft_final.jsonl"
            )
            if (
                path != freeze_path.parent / expected_name
                or artifact["sha256"] != frozen["dataset_files_sha256"][expected_name]
            ):
                raise ValueError("Training artifact must be the approved frozen dataset")
        elif not path.is_relative_to(config.sft_manifest.parent.parent.resolve()):
            raise ValueError("Validation artifact escapes project outputs")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != artifact["sha256"]:
            raise ValueError(f"SFT artifact hash mismatch: {name}")
        rows = [json.loads(line) for line in content.splitlines()]
        if len(rows) != artifact["count"] or not rows:
            raise ValueError(f"Invalid SFT count: {name}")
        for row in rows:
            ids, labels = row["input_ids"], row["labels"]
            if len(ids) > config.max_seq_length:
                raise ValueError(f"{name} exceeds max_seq_length; no truncation is permitted")
            if any(type(i) is not int or i < 0 for i in ids):
                raise ValueError("Invalid token IDs")
            if not 0 < row["prompt_tokens"] < len(ids) or all(v == -100 for v in labels):
                raise ValueError("SFT row lacks assistant supervision")
            if row["attention_mask"] != [1] * len(ids):
                raise ValueError("Prepared SFT rows must be unpadded")
        groups[name] = rows
    provenance = manifest["provenance"]
    validate_sft_pairs(
        groups["english_train"],
        groups.get("multilingual_train", []),
        groups["validation"],
        set(provenance["allowed_train_ids"]),
        set(provenance["all_validation_ids"]) | set(provenance["all_test_ids"]),
        english_only=partial,
        expected_language_counts=manifest.get("approved_language_counts") if schema == 2 else None,
    )
    return groups[config.condition], groups["validation"], manifest


class ResponseCollator:
    """Right-pad without ever regenerating labels or supervising prompt/padding tokens."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def pad(self, rows: list[dict]) -> dict[str, list[list[int]]]:
        length = max(len(row["input_ids"]) for row in rows)
        batch: dict[str, list[list[int]]] = {
            key: [] for key in ("input_ids", "labels", "attention_mask")
        }
        for row in rows:
            n = length - len(row["input_ids"])
            batch["input_ids"].append(row["input_ids"] + [self.pad_token_id] * n)
            batch["labels"].append(row["labels"] + [-100] * n)
            batch["attention_mask"].append(row["attention_mask"] + [0] * n)
        return batch

    def __call__(self, rows: list[dict]) -> dict[str, Any]:
        import torch

        return {key: torch.tensor(value, dtype=torch.long) for key, value in self.pad(rows).items()}
