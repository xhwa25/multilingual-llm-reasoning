"""Guarded, resumable adapter-only supervised training; dry-run never downloads weights."""

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

from filelock import FileLock

from .training_config import TrainingConfig, config_dict
from .training_data import load_training_data
from .training_hardware import probe_hardware


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")
    temporary.replace(path)


def run_identity(config: TrainingConfig, mode: str) -> dict:
    settings = config_dict(config)
    settings.pop("output_dir")
    settings.pop("sft_manifest")  # Manifest hash identifies data across Mac/Colab paths.
    return {
        "config": settings,
        "mode": mode,
        "sft_manifest_sha256": hashlib.sha256(config.sft_manifest.read_bytes()).hexdigest(),
    }


def resolve_resume(directory: Path, resume: str | None, identity: dict) -> Path | None:
    if resume is None:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(
                "Run directory is nonempty; use --resume latest or choose a new output_dir."
            )
        return None
    metadata = directory / "run.json"
    if not metadata.exists() or json.loads(metadata.read_text())["identity"] != identity:
        raise ValueError(
            "Resume config, data hash, condition or mode mismatch; use the original run config."
        )
    if resume == "latest":
        candidates = [
            p
            for p in directory.glob("checkpoint-*")
            if p.is_dir() and p.name.removeprefix("checkpoint-").isdigit()
        ]
        if not candidates:
            raise ValueError("No checkpoints available to resume")
        checkpoint = max(candidates, key=lambda p: int(p.name.split("-")[-1]))
    else:
        checkpoint = Path(resume).resolve()
    if checkpoint.parent.resolve() != directory.resolve():
        raise ValueError("Resume checkpoint must belong to this condition and mode directory")
    required = (
        "adapter_model.safetensors",
        "adapter_config.json",
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    )
    if any(not (checkpoint / file).is_file() for file in required):
        raise ValueError(
            "Incomplete checkpoint; exact resume needs adapters, optimizer, "
            "scheduler, RNG and trainer state. Select an earlier complete checkpoint."
        )
    if identity["config"]["precision"] == "fp16" and not (checkpoint / "scaler.pt").exists():
        raise ValueError("FP16 resume requires scaler.pt")
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if state.get("global_step", 0) >= state.get("max_steps", 0):
        raise ValueError("Checkpoint already finished its training budget; no steps remain")
    return checkpoint


def train_adapter(
    config: TrainingConfig,
    *,
    mode: str = "dry-run",
    confirm_gpu: bool = False,
    resume: str | None = None,
    check_hardware: bool = False,
) -> dict:
    if mode not in ("dry-run", "smoke", "full"):
        raise ValueError("mode must be dry-run, smoke, or full")
    if mode != "dry-run" and not confirm_gpu:
        raise ValueError(
            "GPU training requires explicit confirmation: pass --confirm-gpu-training "
            "only after reviewing the dry-run."
        )
    hardware = (
        probe_hardware(config)
        if (check_hardware or mode != "dry-run")
        else {
            "status": "not_checked",
            "blockers": [],
            "note": (
                "Local validation does not require training packages or CUDA. "
                "Use --check-hardware on Colab."
            ),
        }
    )
    report: dict = {
        "mode": mode,
        "config": config_dict(config),
        "hardware": hardware,
        "blockers": list(hardware["blockers"]),
        "training_started": False,
        "validation_scope": "gpu_preflight" if (check_hardware or mode != "dry-run") else "local",
        "data_status": "missing" if not config.sft_manifest.exists() else "invalid",
    }
    try:
        if config.sft_manifest.exists():
            declared = json.loads(config.sft_manifest.read_text())
            report["export_status"] = declared.get("status", "unknown")
            condition_status = (
                declared.get("conditions", {}).get(config.condition, {}).get("status")
            )
            if condition_status == "missing":
                report["data_status"] = "missing"
            elif declared.get("status") == "preview_not_training_ready":
                report["data_status"] = "unreviewed"
        train, validation, manifest = load_training_data(config)
        report.update(
            data_status="validated",
            paired_data_ready=manifest.get("training_ready", False),
            train_examples=len(train),
            validation_examples=len(validation),
            effective_batch_size=config.batch_size * config.gradient_accumulation_steps,
            optimizer_steps=20
            if mode == "smoke"
            else math.ceil(
                math.ceil(
                    math.ceil(len(train) / config.batch_size) / config.gradient_accumulation_steps
                )
                * config.epochs
            ),
            response_only_loss=True,
            truncation=False,
            question_languages=manifest["diagnostics"][config.condition][
                "question_language_counts"
            ],
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report["blockers"].append(f"SFT validation failed: {exc}")
    report["validation_ready"] = report["data_status"] == "validated"
    report["gpu_ready"] = (
        None if hardware.get("status") == "not_checked" else not hardware["blockers"]
    )
    report["ready"] = not report["blockers"]
    report["training_ready"] = report["validation_ready"] and report["gpu_ready"] is True
    if mode == "dry-run":
        return report
    if report["blockers"]:
        raise ValueError(
            "Training blocked before model download:\n" + "\n".join(report["blockers"])
        )
    directory = config.output_dir / mode
    identity = run_identity(config, mode)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory.parent / f".{mode}.lock"), timeout=0):
        checkpoint = resolve_resume(directory, resume, identity)
        directory.mkdir(exist_ok=True)
        started = time.monotonic()
        run_report = {
            "identity": identity,
            "preflight": report,
            "status": "starting",
            "resume_checkpoint": str(checkpoint) if checkpoint else None,
        }
        write_json(directory / "run.json", run_report)
        try:
            from .training_runtime import execute_training

            metrics = execute_training(
                config, mode, directory, checkpoint, train, validation, manifest
            )
            run_report.update(status="completed", metrics=metrics)
        except Exception as exc:
            run_report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise ValueError(
                f"Adapter training stopped: {exc}. Inspect {directory}/run.json and logs. "
                "For CUDA OOM, free VRAM or reduce batch/length or use QLoRA on a larger GPU. "
                "Use a new run directory if changing settings; no full-fine-tuning "
                "fallback occurred."
            ) from exc
        finally:
            run_report["wall_seconds_this_invocation"] = time.monotonic() - started
            write_json(directory / "run.json", run_report)
        return run_report
