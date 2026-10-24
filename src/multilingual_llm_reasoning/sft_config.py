"""SFT data settings; the only implemented target policy is original English solutions."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from .baseline_config import CHECKPOINT


@dataclass(frozen=True)
class SFTConfig:
    preparation_manifest: Path
    translation_manifest: Path
    output_dir: Path
    tokenizer_checkpoint: str
    tokenizer_revision: str
    train_size: int
    validation_size: int
    seed: int
    max_seq_length: int
    solution_policy: str
    allow_unreviewed_questions: bool


def load_sft_config(path: Path) -> SFTConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    keys = set(SFTConfig.__dataclass_fields__) | {"schema_version"}
    if not isinstance(raw, dict) or set(raw) != keys:
        raise ValueError(f"SFT config requires exactly {sorted(keys)}")
    schema = raw.pop("schema_version")
    if type(schema) is not int or schema != 1:
        raise ValueError("schema_version must be 1")
    for key in ("preparation_manifest", "translation_manifest", "output_dir"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError(f"{key} must be a path")
        raw[key] = (path.resolve().parent / raw[key]).resolve()
    for key in ("train_size", "validation_size", "seed", "max_seq_length"):
        if type(raw[key]) is not int or raw[key] < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid integer for {key}")
    if raw["train_size"] % 4:
        raise ValueError("train_size must be divisible by four for exact language balance")
    if raw["max_seq_length"] > 32768:
        raise ValueError("max_seq_length exceeds Qwen2.5-3B's documented context length")
    if raw["tokenizer_checkpoint"] != CHECKPOINT:
        raise ValueError(f"tokenizer_checkpoint must be {CHECKPOINT}")
    if not isinstance(raw["tokenizer_revision"], str) or not raw["tokenizer_revision"].strip():
        raise ValueError("tokenizer_revision must be a model ref or commit")
    if raw["solution_policy"] != "original_english":
        raise ValueError(
            "Only solution_policy: original_english is implemented; no translated-target claims"
        )
    if type(raw["allow_unreviewed_questions"]) is not bool:
        raise ValueError("allow_unreviewed_questions must be boolean")
    return SFTConfig(**raw)
