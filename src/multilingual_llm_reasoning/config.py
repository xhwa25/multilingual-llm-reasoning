"""Strict, versioned configuration with config-relative paths."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LoRAConfig:
    rank: int
    alpha: int
    dropout: float


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    languages: tuple[str, ...]
    dataset_path: Path
    output_dir: Path
    model: str
    lora: LoRAConfig


def load_config(path: Path) -> ExperimentConfig:
    """Load a strict schema; reject typos rather than silently using defaults."""
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    keys = {"schema_version", "seed", "languages", "dataset_path", "output_dir", "model", "lora"}
    if not isinstance(raw, dict) or set(raw) != keys:
        raise ValueError(f"Config must contain exactly: {sorted(keys)}")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    if type(raw["seed"]) is not int or raw["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    languages = raw["languages"]
    if (
        not isinstance(languages, list)
        or not languages
        or any(not isinstance(x, str) or not x.strip() for x in languages)
        or len(set(languages)) != len(languages)
    ):
        raise ValueError("languages must be unique nonempty strings")
    for key in ("dataset_path", "output_dir", "model"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    lora = raw["lora"]
    if not isinstance(lora, dict) or set(lora) != {"rank", "alpha", "dropout"}:
        raise ValueError("lora requires rank, alpha, dropout")
    for key in ("rank", "alpha"):
        if type(lora[key]) is not int or lora[key] <= 0:
            raise ValueError(f"lora.{key} must be a positive integer")
    if type(lora["dropout"]) not in (int, float) or not 0 <= lora["dropout"] < 1:
        raise ValueError("lora.dropout must be in [0, 1)")
    base = path.resolve().parent
    return ExperimentConfig(
        seed=raw["seed"],
        languages=tuple(languages),
        dataset_path=(base / raw["dataset_path"]).resolve(),
        output_dir=(base / raw["output_dir"]).resolve(),
        model=raw["model"],
        lora=LoRAConfig(**lora),
    )
