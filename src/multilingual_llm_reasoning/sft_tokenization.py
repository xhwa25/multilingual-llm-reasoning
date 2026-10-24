"""Qwen tokenizer only: no weights, model loading, training or GPU dependency."""

import hashlib
import re
from importlib.metadata import version
from typing import Any, Protocol

from .sft_config import SFTConfig


class ChatTokenizer(Protocol):
    eos_token_id: int

    def get_chat_template(self) -> str: ...
    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> Any: ...
    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]: ...


def load_sft_tokenizer(config: SFTConfig) -> tuple[ChatTokenizer, dict[str, Any]]:
    try:
        from huggingface_hub import HfApi
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ValueError("Install tokenizer support: pip install -e '.[sft]'") from exc
    revision = config.tokenizer_revision
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        revision = HfApi().model_info(config.tokenizer_checkpoint, revision=revision).sha
    if not revision or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Could not resolve tokenizer commit")
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_checkpoint, revision=revision, trust_remote_code=False
    )
    template = tokenizer.get_chat_template()
    if not template or tokenizer.eos_token_id is None:
        raise ValueError("A chat template and EOS token are required")
    return tokenizer, {
        "checkpoint": config.tokenizer_checkpoint,
        "requested_revision": config.tokenizer_revision,
        "resolved_revision": revision,
        "chat_template": template,
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "transformers_version": version("transformers"),
        "eos_token_id": tokenizer.eos_token_id,
    }


def tokenize_sft(tokenizer: ChatTokenizer, messages: list[dict[str, str]]) -> dict[str, Any]:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    direct_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if ids != direct_ids:
        raise ValueError("Rendered and directly templated tokenization disagree")
    prefix = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    prefix_ids = tokenizer(prefix, add_special_tokens=False, truncation=False)["input_ids"]
    if not text.startswith(prefix) or ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("Assistant prefix is not token-aligned; refusing an incorrect loss mask")
    if len(ids) <= len(prefix_ids) or tokenizer.eos_token_id not in ids[len(prefix_ids) :]:
        raise ValueError("Assistant target is empty or lacks its terminating EOS token")
    return {
        "formatted_text": text,
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": [-100] * len(prefix_ids) + ids[len(prefix_ids) :],
        "token_length": len(ids),
        "prompt_tokens": len(prefix_ids),
        "target_tokens": len(ids) - len(prefix_ids),
    }
