"""Heavy dependencies are imported only after confirmation and successful preflight."""

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

from .logging import configure_logging
from .sft_tokenization import tokenize_sft
from .training_config import TrainingConfig
from .training_data import ResponseCollator


def assert_finite_losses(metrics: dict) -> None:
    """Reject non-finite train/evaluation losses before reporting success."""
    for key, value in metrics.items():
        if "loss" in key:
            try:
                finite = math.isfinite(float(value))
            except (TypeError, ValueError, OverflowError):
                finite = False
            if not finite:
                raise ValueError(f"Non-finite training metric: {key}={value}")


def trainable_digest(model: Any) -> str:
    """Hash trainable adapter tensors to prove the optimizer changed parameters."""
    checksum = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            checksum.update(name.encode())
            checksum.update(parameter.detach().float().cpu().numpy().tobytes())
    return checksum.hexdigest()


def verify_smoke_checkpoint(directory: Path, precision: str) -> None:
    """Deserialize adapter and optimizer state from the last smoke checkpoint."""
    import torch
    from safetensors.torch import load_file

    checkpoint = directory / "checkpoint-20"
    required = (
        "adapter_model.safetensors",
        "adapter_config.json",
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    )
    if any(not (checkpoint / name).is_file() for name in required):
        raise ValueError("20-step smoke checkpoint is incomplete")
    if precision == "fp16" and not (checkpoint / "scaler.pt").is_file():
        raise ValueError("FP16 smoke checkpoint lacks scaler state")
    adapter = load_file(checkpoint / "adapter_model.safetensors", device="cpu")
    optimizer = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
    scheduler = torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=True)
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if not adapter or not optimizer.get("state") or not scheduler or state.get("global_step") != 20:
        raise ValueError("Smoke checkpoint cannot be loaded or has wrong step/state")


def assert_adapter_only(model: Any) -> dict:
    trainable = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise ValueError("Safety check failed: only LoRA parameters may be trainable")
    return {
        "trainable_parameters": sum(count for _, count in trainable),
        "trainable_parameter_names": [name for name, _ in trainable],
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }


def execute_training(
    config: TrainingConfig,
    mode: str,
    directory: Path,
    checkpoint: Path | None,
    training: list[dict],
    validation: list[dict],
    manifest: dict,
) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainerCallback,
        set_seed,
    )
    from trl import SFTConfig, SFTTrainer

    from .training import write_json

    set_seed(config.seed)
    torch.cuda.set_device(config.device_index)
    torch.cuda.reset_peak_memory_stats(config.device_index)
    started = time.monotonic()
    logger = configure_logging()
    tokenizer = AutoTokenizer.from_pretrained(
        config.model, revision=config.revision, trust_remote_code=False
    )
    template_hash = hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest()
    if template_hash != manifest["tokenizer"]["chat_template_sha256"]:
        raise ValueError("Tokenizer chat template differs from SFT provenance")
    for row in training + validation:
        actual = tokenize_sft(tokenizer, row["messages"])
        if any(actual[key] != row[key] for key in ("input_ids", "labels", "attention_mask")):
            raise ValueError("SFT token IDs or assistant mask do not match the pinned tokenizer")
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if config.precision == "bf16" else torch.float16
    kwargs: dict = {
        "revision": config.revision,
        "trust_remote_code": False,
        "torch_dtype": dtype,
        "attn_implementation": "sdpa",
        "device_map": {"": config.device_index},
    }
    if config.qlora:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    if config.qlora:
        import bitsandbytes as bnb

        try:
            probe = torch.ones((64, 64), device=f"cuda:{config.device_index}", dtype=dtype)
            bnb.functional.quantize_4bit(probe, quant_type="nf4")
            del probe
        except Exception as exc:
            raise ValueError(
                "bitsandbytes CUDA kernel probe failed; reinstall a compatible wheel"
            ) from exc
    model = AutoModelForCausalLM.from_pretrained(config.model, **kwargs)
    if config.qlora:
        if not getattr(model, "is_loaded_in_4bit", False):
            raise ValueError("Requested QLoRA but model is not 4-bit; refusing fallback")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    model.config.use_cache = False
    modules = {name.split(".")[-1] for name, _ in model.named_modules()}
    if not set(config.target_modules) <= modules:
        raise ValueError("Configured LoRA target modules are missing in model")
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            bias="none",
            modules_to_save=None,
        ),
    )
    parameter_report = assert_adapter_only(model)
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise ValueError(
            "CPU/disk offload detected; this backend requires a fully CUDA-resident model"
        )

    class AdapterTrainer(SFTTrainer):
        def _save(self, output_dir: str | None = None, state_dict: Any = None) -> None:
            # Never serialize the frozen base weights or embeddings, even in checkpoints.
            target = Path(output_dir or self.args.output_dir)
            target.mkdir(parents=True, exist_ok=True)
            assert_adapter_only(self.model)
            self.model.save_pretrained(target, safe_serialization=True, save_embedding_layers=False)
            tokenizer.save_pretrained(target)
            write_json(target / "training_args.json", self.args.to_dict())

    class ResourceLogger(TrainerCallback):
        def on_log(
            self, args: Any, state: Any, control: Any, logs: dict | None = None, **kwargs: Any
        ) -> None:
            assert_finite_losses(logs or {})
            event = {
                "step": state.global_step,
                "epoch": state.epoch,
                "elapsed_seconds": time.monotonic() - started,
                "gpu_allocated_gib": torch.cuda.memory_allocated() / 2**30,
                "gpu_reserved_gib": torch.cuda.memory_reserved() / 2**30,
                "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                **(logs or {}),
            }
            with (directory / "training_log.jsonl").open("a") as handle:
                handle.write(json.dumps(event, default=str) + "\n")
            logger.info("training_progress", extra={"details": event})

    smoke = mode == "smoke"
    args = SFTConfig(
        output_dir=str(directory),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.scheduler,
        num_train_epochs=config.epochs,
        max_steps=20 if smoke else -1,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        seed=config.seed,
        data_seed=config.seed,
        bf16=config.precision == "bf16",
        fp16=config.precision == "fp16",
        tf32=False,
        optim="adamw_torch",
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1 if smoke else config.logging_steps,
        eval_strategy="steps",
        eval_steps=5 if smoke else config.eval_steps,
        save_strategy="steps",
        save_steps=5 if smoke else config.save_steps,
        save_total_limit=config.save_total_limit,
        save_only_model=False,
        save_safetensors=True,
        report_to="none",
        push_to_hub=False,
        max_length=config.max_seq_length,
        packing=False,
        padding_free=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        # Labels have already been audited. Qwen's template lacks TRL generation masks.
        assistant_only_loss=False,
        completion_only_loss=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        prediction_loss_only=True,
        eval_accumulation_steps=1,
    )
    keys = ("input_ids", "attention_mask", "labels")
    trainer = AdapterTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        data_collator=ResponseCollator(tokenizer.pad_token_id),
        train_dataset=Dataset.from_list([{k: row[k] for k in keys} for row in training]),
        eval_dataset=Dataset.from_list([{k: row[k] for k in keys} for row in validation]),
        callbacks=[ResourceLogger()],
    )
    write_json(directory / "adapter_parameters.json", parameter_report)
    # Evaluating before training exercises memory and reports a common starting loss.
    initial = trainer.evaluate(metric_key_prefix="initial_eval")
    assert_finite_losses(initial)
    initial_adapter_digest = trainable_digest(model)
    result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    final = trainer.evaluate()
    assert_finite_losses(result.metrics)
    assert_finite_losses(final)
    if smoke and trainer.state.global_step != 20:
        raise ValueError("Smoke run did not complete exactly 20 optimizer steps")
    if smoke and trainable_digest(model) == initial_adapter_digest:
        raise ValueError("Smoke optimizer steps did not change LoRA adapter parameters")
    if smoke:
        verify_smoke_checkpoint(directory, config.precision)
    trainer.save_model(str(directory / "adapter"))
    from .training_config import config_dict

    budget = config_dict(config)
    for key in ("name", "condition", "output_dir", "sft_manifest"):
        budget.pop(key)
    write_json(
        directory / "adapter/evaluation_provenance.json",
        {
            "schema_version": 1,
            "model": config.model,
            "revision": config.revision,
            "condition": config.condition,
            "train_ids": sorted(manifest["selected_train_ids"]),
            "validation_ids": sorted(manifest["selected_validation_ids"]),
            "training_budget": {**budget, "mode": mode, "global_step": trainer.state.global_step},
        },
    )
    trainer.save_state()
    torch.cuda.synchronize()
    metrics = {
        **initial,
        **result.metrics,
        **final,
        **parameter_report,
        "global_step": trainer.state.global_step,
        "runtime_seconds": time.monotonic() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "smoke_checkpoint_load_verified": smoke,
        "smoke_optimizer_update_verified": smoke,
    }
    write_json(directory / "metrics.json", metrics)
    return metrics
