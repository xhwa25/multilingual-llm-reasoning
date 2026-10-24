"""Lazy local inference adapters. No model dependencies are needed for mock runs."""

import hashlib
import re
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any, Protocol

from .baseline_config import BaselineConfig


@dataclass(frozen=True)
class Generation:
    prediction: str
    finish_reason: str
    generated_tokens: int | None = None


class BaselineBackend(Protocol):
    metadata: dict[str, Any]

    def generate(self, messages: list[list[dict[str, str]]]) -> list[Generation]: ...


def resolve_revision(config: BaselineConfig) -> str:
    if config.backend == "mock":
        return "mock-baseline-v1"
    if re.fullmatch(r"[a-f0-9]{40}", config.revision):
        return config.revision
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ValueError("Install model dependencies with pip install -e '.[inference]'") from exc
    revision = HfApi().model_info(config.model, revision=config.revision).sha
    if not revision or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Could not resolve an immutable model commit")
    return revision


def select_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested.startswith("cuda"):
        index = int(requested.split(":")[1]) if ":" in requested else 0
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise ValueError(f"Requested CUDA device unavailable: {requested}")
        return f"cuda:{index}"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Requested MPS device unavailable")
    return requested


def select_dtype(torch: Any, requested: str, device: str) -> str:
    if requested == "auto":
        if device == "cpu":
            return "float32"
        if device.startswith("cuda") and torch.cuda.is_bf16_supported():
            return "bfloat16"
        return "float16"
    if device == "cpu" and requested != "float32":
        raise ValueError("This CPU adapter requires float32; use dtype: auto or float32")
    if device == "mps" and requested == "bfloat16":
        raise ValueError("Use float16 or float32 on MPS for this adapter")
    return requested


def _metadata(
    config: BaselineConfig,
    revision: str,
    device: str,
    dtype: str,
    template: str,
    versions: dict[str, str],
) -> dict[str, Any]:
    return {
        "backend": config.backend,
        "model_checkpoint": config.model,
        "model_revision": revision,
        "device": device,
        "dtype": dtype,
        "chat_template": template,
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "versions": versions,
    }


class MockBaselineBackend:
    def __init__(self, config: BaselineConfig, revision: str) -> None:
        self.metadata = _metadata(config, revision, "cpu", "none", "mock-messages-v1", {})
        self.metadata["model_checkpoint"] = "mock-baseline-v1"

    def generate(self, messages: list[list[dict[str, str]]]) -> list[Generation]:
        # Deliberately unrelated to the gold answers. Never claims to be Qwen inference.
        return [Generation("FINAL_ANSWER: 0", "mock", None) for _ in messages]


class TransformersBackend:
    def __init__(self, config: BaselineConfig, revision: str) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        except ImportError as exc:
            raise ValueError(
                "Install model dependencies with pip install -e '.[inference]'"
            ) from exc
        self.torch = torch
        self.device = select_device(torch, config.device)
        if self.device.startswith("cuda"):
            torch.cuda.set_device(self.device)
        self.dtype = select_dtype(torch, config.dtype, self.device)
        torch.manual_seed(config.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model, revision=revision, trust_remote_code=False, padding_side="left"
        )
        template = self.tokenizer.get_chat_template()
        if not template:
            raise ValueError("Checkpoint tokenizer has no chat template")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model,
            revision=revision,
            trust_remote_code=False,
            torch_dtype=getattr(torch, self.dtype),
        ).to(self.device)
        self.model.eval()
        self.max_new_tokens = config.max_new_tokens
        self.eos = self.model.generation_config.eos_token_id
        # Do not inherit checkpoint sampling defaults. Temperature=None because HF
        # greedy generation ignores temperature and rejects zero in some versions.
        self.generation_config = GenerationConfig(
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            top_k=None,
            max_new_tokens=config.max_new_tokens,
            repetition_penalty=1.0,
            eos_token_id=self.eos,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            use_cache=True,
        )
        self.metadata = _metadata(
            config,
            revision,
            self.device,
            self.dtype,
            template,
            {name: version(name) for name in ("torch", "transformers")},
        )

    def _sync(self) -> None:
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)
        elif self.device == "mps":
            self.torch.mps.synchronize()

    def generate(self, messages: list[list[dict[str, str]]]) -> list[Generation]:
        prompts = [
            self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            for conversation in messages
        ]
        inputs = self.tokenizer(
            prompts, padding=True, truncation=False, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        width = inputs["input_ids"].shape[1]
        context_limit = self.model.config.max_position_embeddings
        if width + self.max_new_tokens > context_limit:
            raise ValueError("context_length_exceeded: inputs are never silently truncated")
        self._sync()
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, generation_config=self.generation_config)
        self._sync()
        # Slice the common padded input width, not each example's unpadded length.
        continuations = generated[:, width:]
        texts = self.tokenizer.batch_decode(continuations, skip_special_tokens=True)
        eos = set(self.eos if isinstance(self.eos, list) else [self.eos])
        results = []
        for text, tokens in zip(texts, continuations.tolist(), strict=True):
            stop = next((i for i, token in enumerate(tokens) if token in eos), None)
            count = stop + 1 if stop is not None else len(tokens)
            results.append(Generation(text, "eos" if stop is not None else "length", count))
        return results


class VLLMBackend:
    def __init__(self, config: BaselineConfig, revision: str) -> None:
        try:
            import torch
            from transformers import GenerationConfig
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ValueError("Install vLLM separately on a supported CUDA host") from exc
        device = select_device(torch, config.device)
        if device != "cuda:0":
            raise ValueError(
                "vLLM requires visible CUDA GPU 0; select hardware via CUDA_VISIBLE_DEVICES"
            )
        torch.cuda.set_device(device)
        dtype = select_dtype(torch, config.dtype, device)
        self.model = LLM(
            model=config.model,
            revision=revision,
            tokenizer_revision=revision,
            trust_remote_code=False,
            dtype=dtype,
            seed=config.seed,
            tensor_parallel_size=1,
            generation_config="vllm",
        )
        tokenizer = self.model.get_tokenizer()
        template = tokenizer.get_chat_template()
        if not template:
            raise ValueError("Checkpoint tokenizer has no chat template")
        eos = GenerationConfig.from_pretrained(config.model, revision=revision).eos_token_id
        stop_ids = eos if isinstance(eos, list) else [eos]
        self.sampling = SamplingParams(
            stop_token_ids=stop_ids,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            repetition_penalty=1.0,
            max_tokens=config.max_new_tokens,
            seed=config.seed,
            n=1,
        )
        self.metadata = _metadata(
            config,
            revision,
            device,
            dtype,
            template,
            {name: version(name) for name in ("torch", "vllm")},
        )

    def generate(self, messages: list[list[dict[str, str]]]) -> list[Generation]:
        outputs = self.model.chat(
            messages, sampling_params=self.sampling, add_generation_prompt=True, use_tqdm=False
        )
        # LLM.chat uses the checkpoint tokenizer's template and returns request order.
        return [
            Generation(
                result.outputs[0].text,
                result.outputs[0].finish_reason,
                len(result.outputs[0].token_ids),
            )
            for result in outputs
        ]


def create_backend(config: BaselineConfig, revision: str) -> BaselineBackend:
    cls = {"mock": MockBaselineBackend, "transformers": TransformersBackend, "vllm": VLLMBackend}
    return cls[config.backend](config, revision)
