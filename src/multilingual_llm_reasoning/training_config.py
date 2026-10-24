"""Strict, reproducible single-GPU adapter-training configuration."""

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TrainingConfig:
    name: str
    sft_manifest: Path
    condition: str
    output_dir: Path
    model: str
    revision: str
    qlora: bool
    precision: str
    device_index: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    max_seq_length: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    scheduler: str
    epochs: float
    seed: int
    warmup_ratio: float
    weight_decay: float
    logging_steps: int
    eval_steps: int
    save_steps: int
    save_total_limit: int
    gradient_checkpointing: bool
    min_free_vram_gib: float


def load_training_config(path: Path) -> TrainingConfig:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or set(raw) != set(TrainingConfig.__dataclass_fields__) | {
        "schema_version"
    }:
        raise ValueError("Training config has missing or unknown fields")
    schema = raw.pop("schema_version")
    if type(schema) is not int or schema != 1:
        raise ValueError("Training schema_version must be 1")
    for key in ("sft_manifest", "output_dir"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError(f"{key} must be a nonempty path")
        raw[key] = (path.resolve().parent / raw[key]).resolve()
    for key in (
        "device_index",
        "lora_rank",
        "lora_alpha",
        "max_seq_length",
        "batch_size",
        "gradient_accumulation_steps",
        "seed",
        "logging_steps",
        "eval_steps",
        "save_steps",
        "save_total_limit",
    ):
        if type(raw[key]) is not int or raw[key] < (0 if key in ("seed", "device_index") else 1):
            raise ValueError(f"Invalid integer {key}")
    for key in (
        "lora_dropout",
        "learning_rate",
        "epochs",
        "warmup_ratio",
        "weight_decay",
        "min_free_vram_gib",
    ):
        if type(raw[key]) not in (int, float) or not math.isfinite(raw[key]) or raw[key] < 0:
            raise ValueError(f"Invalid finite number {key}")
    if not 0 < raw["learning_rate"] < 1 or raw["epochs"] <= 0 or raw["min_free_vram_gib"] <= 0:
        raise ValueError("learning_rate, epochs and minimum VRAM must be positive")
    if raw["lora_dropout"] >= 1 or raw["warmup_ratio"] >= 1:
        raise ValueError("dropout and warmup_ratio must be below 1")
    for key in ("qlora", "gradient_checkpointing"):
        if type(raw[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    modules = raw["target_modules"]
    allowed = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    if not isinstance(modules, list) or not modules or any(m not in allowed for m in modules):
        raise ValueError("target_modules must name Qwen attention/MLP projection modules")
    if len(set(modules)) != len(modules):
        raise ValueError("Duplicate target_modules")
    if raw["model"] != "Qwen/Qwen2.5-3B-Instruct" or not re.fullmatch(
        r"[a-f0-9]{40}", str(raw["revision"])
    ):
        raise ValueError("Use Qwen/Qwen2.5-3B-Instruct with a pinned 40-character commit")
    if raw["condition"] not in ("english_train", "multilingual_train"):
        raise ValueError("condition must be english_train or multilingual_train")
    if raw["precision"] not in ("bf16", "fp16") or raw["scheduler"] not in (
        "linear",
        "cosine",
        "constant",
    ):
        raise ValueError("Unsupported precision or scheduler")
    if not isinstance(raw["name"], str) or not re.fullmatch(r"[a-z0-9_-]+", raw["name"]):
        raise ValueError("name must be a simple run identifier")
    if raw["max_seq_length"] > 32768:
        raise ValueError("Sequence length exceeds Qwen context")
    return TrainingConfig(**raw)


def config_dict(config: TrainingConfig) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }
