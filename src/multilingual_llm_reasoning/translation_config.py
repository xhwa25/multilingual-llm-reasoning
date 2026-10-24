"""Strict translation settings; prices are user-supplied, never assumed current."""

import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import yaml

from .translation import PROMPT_VERSION


@dataclass(frozen=True)
class TranslationConfig:
    manifest_path: Path
    output_dir: Path
    splits: tuple[str, ...]
    provider: str
    model: str
    prompt_version: str
    seed: int
    review_sample_size: int
    max_requests: int
    max_cost_usd: Decimal
    input_usd_per_million: Decimal | None
    output_usd_per_million: Decimal | None
    max_output_tokens: int
    max_attempts: int
    backoff_seconds: float
    max_backoff_seconds: float
    timeout_seconds: float


def load_translation_config(path: Path) -> TranslationConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    keys = set(TranslationConfig.__dataclass_fields__) | {"schema_version"}
    if not isinstance(raw, dict) or set(raw) != keys:
        raise ValueError(f"Translation config requires exactly {sorted(keys)}")
    schema_version = raw.pop("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError("schema_version must be 1")
    if raw["prompt_version"] != PROMPT_VERSION:
        raise ValueError(f"prompt_version must be {PROMPT_VERSION}")
    for key in ("manifest_path", "output_dir"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError(f"{key} must be a nonempty path")
        raw[key] = (path.resolve().parent / raw[key]).resolve()
    splits = raw["splits"]
    if (
        not isinstance(splits, list)
        or not splits
        or any(x not in ("train", "validation", "test") for x in splits)
        or len(set(splits)) != len(splits)
    ):
        raise ValueError("splits must contain unique train/validation/test names")
    raw["splits"] = tuple(splits)
    if raw["provider"] not in ("mock", "openai", "local_transformers"):
        raise ValueError("provider must be mock, openai, or local_transformers")
    if not isinstance(raw["model"], str) or not raw["model"].strip():
        raise ValueError("model must be a nonempty identifier")
    if raw["provider"] == "mock" and raw["model"] != "mock-v1":
        raise ValueError("mock provider requires model: mock-v1")
    if (
        raw["provider"] == "local_transformers"
        and raw["model"] != "facebook/nllb-200-distilled-600M"
    ):
        raise ValueError("local_transformers currently supports facebook/nllb-200-distilled-600M")
    for key in ("seed", "review_sample_size", "max_requests", "max_output_tokens", "max_attempts"):
        if type(raw[key]) is not int or raw[key] < (0 if key in ("seed", "max_requests") else 1):
            raise ValueError(f"invalid nonnegative/positive integer: {key}")
    if not 20 <= raw["review_sample_size"] <= 30:
        raise ValueError("review_sample_size must be 20–30 original questions")
    if raw["max_output_tokens"] < 16 or raw["max_attempts"] > 10:
        raise ValueError("max_output_tokens must be >=16 and max_attempts <=10")
    for key in ("backoff_seconds", "max_backoff_seconds", "timeout_seconds"):
        value = raw[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if raw["timeout_seconds"] <= 0 or raw["max_backoff_seconds"] < raw["backoff_seconds"]:
        raise ValueError("timeout must be positive; max backoff must be >= initial backoff")
    for key in ("max_cost_usd", "input_usd_per_million", "output_usd_per_million"):
        value = raw[key]
        if value is None and key != "max_cost_usd":
            continue
        try:
            amount = Decimal(str(value))
        except InvalidOperation as exc:
            raise ValueError(f"{key} must be a nonnegative decimal") from exc
        if not amount.is_finite() or amount < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
        raw[key] = amount
    return TranslationConfig(**raw)
