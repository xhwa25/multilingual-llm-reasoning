"""Fixed cross-language baseline settings and small-run defaults."""

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from .translation import LANGUAGES

CHECKPOINT = "Qwen/Qwen2.5-3B-Instruct"
PROMPT_VERSION = "baseline-final-answer-v1"
# Same English instruction for every language: no per-language adaptations.
INSTRUCTION = (
    "Solve the problem and give a concise answer. End your response with a separate line "
    "in exactly this format: FINAL_ANSWER: <number>. Replace <number> with a single signed "
    "decimal number using a dot for the decimal point, without units or surrounding text. "
    "Do not write anything after that final line."
)


@dataclass(frozen=True)
class BaselineConfig:
    translation_manifest: Path
    output_dir: Path
    backend: str
    model: str
    revision: str
    device: str
    dtype: str
    batch_size: int
    max_new_tokens: int
    temperature: int | float
    do_sample: bool
    seed: int
    num_questions: int
    split: str
    prompt_version: str
    allow_unreviewed_translations: bool
    languages: tuple[str, ...] = LANGUAGES


def load_baseline_config(path: Path) -> BaselineConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    keys = set(BaselineConfig.__dataclass_fields__) | {"schema_version"}
    if isinstance(raw, dict):
        raw.setdefault("languages", list(LANGUAGES))
    if not isinstance(raw, dict) or set(raw) != keys:
        raise ValueError(f"Baseline config requires exactly {sorted(keys)}")
    schema = raw.pop("schema_version")
    if type(schema) is not int or schema != 1:
        raise ValueError("schema_version must be 1")
    if raw["languages"] not in (["en"], list(LANGUAGES)):
        raise ValueError("languages must be [en] for source-only smoke or [en, zh, de, ja]")
    raw["languages"] = tuple(raw["languages"])
    for key in ("translation_manifest", "output_dir"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError(f"{key} must be a path")
        raw[key] = (path.resolve().parent / raw[key]).resolve()
    if raw["backend"] not in ("mock", "transformers", "vllm"):
        raise ValueError("backend must be mock, transformers, or vllm")
    if raw["model"] != CHECKPOINT or raw["prompt_version"] != PROMPT_VERSION:
        raise ValueError(f"Use model {CHECKPOINT} and prompt_version {PROMPT_VERSION}")
    if not isinstance(raw["revision"], str) or not raw["revision"].strip():
        raise ValueError("revision must be a nonempty model ref or commit SHA")
    for key in ("batch_size", "max_new_tokens", "num_questions", "seed"):
        if type(raw[key]) is not int or raw[key] < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid integer for {key}")
    if raw["max_new_tokens"] > 8192:
        raise ValueError("max_new_tokens exceeds the model's documented generation limit")
    if type(raw["do_sample"]) is not bool or raw["do_sample"]:
        raise ValueError("All baselines must use do_sample: false")
    if type(raw["temperature"]) not in (int, float) or raw["temperature"] != 0:
        raise ValueError("All baselines must use temperature: 0 (greedy)")
    if type(raw["allow_unreviewed_translations"]) is not bool:
        raise ValueError("allow_unreviewed_translations must be a boolean")
    if raw["split"] not in ("train", "validation", "test"):
        raise ValueError("Invalid split")
    if raw["dtype"] not in ("auto", "float32", "float16", "bfloat16"):
        raise ValueError("dtype must be auto, float32, float16, or bfloat16")
    device = raw["device"]
    if not isinstance(device, str) or not (
        device in ("auto", "cpu", "mps", "cuda")
        or (device.startswith("cuda:") and device[5:].isdigit())
    ):
        raise ValueError("device must be auto, cpu, mps, cuda, or cuda:N")
    if raw["backend"] == "vllm" and device not in ("auto", "cuda", "cuda:0"):
        raise ValueError("This vLLM adapter supports one visible CUDA GPU (auto/cuda/cuda:0)")
    return BaselineConfig(**raw)


def make_messages(question: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": question}]


def decoding_settings(config: BaselineConfig) -> dict[str, object]:
    return {
        "do_sample": False,
        "temperature": 0,
        "num_beams": 1,
        "max_new_tokens": config.max_new_tokens,
        "seed": config.seed,
        "repetition_penalty": 1.0,
        "top_p": 1.0,
    }


def baseline_config_dict(config: BaselineConfig) -> dict[str, object]:
    """Keep legacy four-language cache contracts unchanged; bind explicit English scope."""
    result = asdict(config)
    if config.languages == LANGUAGES:
        result.pop("languages")
    return result
