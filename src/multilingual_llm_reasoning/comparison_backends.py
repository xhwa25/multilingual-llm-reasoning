"""Fresh same-revision base for each condition; explicit PEFT activation audit."""

import hashlib
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from .baseline_backends import TransformersBackend
from .baseline_config import BaselineConfig


def audit_adapter(path: Path, condition: str, config: BaselineConfig, test_ids: set[str]) -> dict:
    required = ("adapter_config.json", "adapter_model.safetensors", "evaluation_provenance.json")
    if any(not (path / name).is_file() for name in required):
        raise ValueError(
            f"Missing adapter/provenance for {condition}: {path}; train/export it first"
        )
    meta = json.loads((path / "evaluation_provenance.json").read_text())
    adapter = json.loads((path / "adapter_config.json").read_text())
    expected = "english_train" if condition == "english_lora" else "multilingual_train"
    if (
        meta.get("condition") != expected
        or meta.get("model") != config.model
        or meta.get("revision") != config.revision
    ):
        raise ValueError("Adapter condition/base revision provenance mismatch")
    if adapter.get("peft_type") != "LORA" or adapter.get("base_model_name_or_path") != config.model:
        raise ValueError("Adapter must be LoRA on the configured base")
    train, validation = set(meta["train_ids"]), set(meta["validation_ids"])
    if not train or not validation or train & validation or (train | validation) & test_ids:
        raise ValueError("Adapter train/validation/test leakage or missing IDs")
    if adapter.get("revision") not in (None, config.revision):
        raise ValueError("Adapter config revision mismatch")
    hashes = {}
    for name in required:
        digest = hashlib.sha256()
        with (path / name).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[name] = digest.hexdigest()
    return {"files": hashes, "provenance": meta}


def activate_adapter(model, path: Path, condition: str):
    from peft import PeftModel

    wrapped = PeftModel.from_pretrained(
        model, str(path), adapter_name=condition, is_trainable=False
    )
    wrapped.set_adapter(condition)
    status = wrapped.get_model_status()
    if (
        status.enabled is not True
        or status.active_adapters != [condition]
        or status.merged_adapters
    ):
        raise ValueError("Adapter activation is disabled, irregular, wrong or merged")
    if status.num_adapter_layers <= 0 or set(wrapped.peft_config) != {condition}:
        raise ValueError("Expected exactly one loaded adapter with nonzero adapter layers")
    wrapped.requires_grad_(False)
    wrapped.eval()
    return wrapped, asdict(status)


class AdapterBackend(TransformersBackend):
    def __init__(self, config: BaselineConfig, revision: str, condition: str, path: Path | None):
        super().__init__(config, revision)
        if path is not None:
            self.model, activation = activate_adapter(self.model, path, condition)
            self.metadata["adapter_activation"] = activation
            self.metadata["versions"]["peft"] = version("peft")
        else:
            if getattr(self.model, "peft_config", None):
                raise ValueError("Base condition unexpectedly has adapters")
            self.metadata["adapter_activation"] = {"active_adapters": [], "untouched_base": True}
        self.metadata["condition_id"] = condition
