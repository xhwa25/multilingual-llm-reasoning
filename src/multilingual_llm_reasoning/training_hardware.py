"""No-download compatibility and conservative single-CUDA-device admission checks."""

import importlib
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

from .training_config import TrainingConfig

# This adapter is deliberately tied to a known TRL/Transformers API family.
PACKAGE_RANGES = {
    "torch": ("2.6", "2.9"),
    "transformers": ("4.56.2", "4.58"),
    "peft": ("0.17.1", "0.18"),
    "trl": ("0.23.1", "0.24"),
    "accelerate": ("1.10", "1.11"),
    "datasets": ("4.0", "4.2"),
}


def required_vram_gib(config: TrainingConfig) -> float:
    """Admission estimate, not an OOM guarantee; includes logits and activation headroom."""
    base = 6.0 if config.qlora else 12.0
    activation = 2.0 * config.batch_size * (config.max_seq_length / 1024)
    if not config.gradient_checkpointing:
        activation *= 3
    adapters = 0.5 * (config.lora_rank / 16) * (len(config.target_modules) / 7)
    return max(config.min_free_vram_gib, base + activation + adapters)


def probe_hardware(config: TrainingConfig) -> dict:
    from packaging.version import Version

    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {},
        "blockers": [],
        "required_free_vram_gib": required_vram_gib(config),
    }
    blockers = report["blockers"]
    if platform.system() != "Linux":
        blockers.append(
            "GPU training profile requires Linux/Colab NVIDIA CUDA. Local dry-run works without it."
        )
    if not (3, 11) <= sys.version_info[:2] <= (3, 12):
        blockers.append(
            "Use the isolated Python 3.11 Colab setup; this GPU profile supports Python 3.11/3.12."
        )
    ranges = dict(PACKAGE_RANGES)
    if config.qlora:
        ranges["bitsandbytes"] = ("0.47", "0.49")
    for package, (minimum, maximum) in ranges.items():
        try:
            installed = version(package)
            report["packages"][package] = installed
            if not Version(minimum) <= Version(installed) < Version(maximum):
                blockers.append(
                    f"{package} {installed} unsupported; require "
                    f">={minimum},<{maximum}. Install .[train] in a fresh Python "
                    f"3.11/3.12 CUDA environment."
                )
        except PackageNotFoundError:
            blockers.append(f"Missing {package}; install .[train] (and .[qlora] for 4-bit).")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        blockers.append(
            "Only single-GPU training is supported; run python directly, not distributed torchrun."
        )
    try:
        import torch

        report["cuda_available"] = torch.cuda.is_available()
        report["torch_cuda_version"] = torch.version.cuda
        if not torch.cuda.is_available() or torch.version.cuda is None:
            blockers.append(
                "NVIDIA CUDA is required. Use a CUDA-enabled PyTorch build on a "
                "Colab/Linux GPU runtime; CPU/MPS fallback is disabled."
            )
        elif torch.cuda.device_count() != 1:
            blockers.append(
                "Expose exactly one GPU using CUDA_VISIBLE_DEVICES=<physical index>; "
                "set device_index: 0. Automatic multi-GPU replication is disabled."
            )
        elif config.device_index >= torch.cuda.device_count():
            blockers.append("device_index is not visible; check CUDA_VISIBLE_DEVICES.")
        else:
            torch.cuda.set_device(config.device_index)
            free, total = torch.cuda.mem_get_info(config.device_index)
            capability = torch.cuda.get_device_capability(config.device_index)
            report.update(
                device=torch.cuda.get_device_name(config.device_index),
                device_index=config.device_index,
                capability=list(capability),
                free_vram_gib=free / 2**30,
                total_vram_gib=total / 2**30,
            )
            if config.precision == "bf16" and (
                capability[0] < 8 or not torch.cuda.is_bf16_supported()
            ):
                blockers.append(
                    "BF16 needs Ampere or newer; use precision: fp16 on a T4, "
                    "identically in both configs."
                )
            if capability < (7, 0):
                blockers.append("This training backend requires CUDA compute capability >=7.0.")
            if free / 2**30 < report["required_free_vram_gib"]:
                blockers.append(
                    "Insufficient free VRAM for conservative estimate. Free GPU "
                    "memory, use QLoRA, reduce batch size/sequence length, or choose "
                    "a larger GPU; regenerate SFT if shortening sequences. Apply "
                    "changes equally to both conditions."
                )
    except (ImportError, OSError, RuntimeError) as exc:
        blockers.append(
            f"Cannot initialize PyTorch/CUDA: {exc}. Use the documented CUDA environment."
        )
    # Import compatibility is checked before model downloads, even if CUDA is unavailable.
    if not any("Missing " in b or "unsupported;" in b for b in blockers):
        try:
            for module in ("transformers", "peft", "trl", "accelerate", "datasets"):
                importlib.import_module(module)
            from trl import SFTConfig, SFTTrainer

            if "dataset_kwargs" not in SFTConfig.__dataclass_fields__ or not SFTTrainer:
                raise RuntimeError("TRL skip_prepare_dataset support is required")
            if config.qlora:
                importlib.import_module("bitsandbytes")
        except Exception as exc:
            blockers.append(
                f"Training package import/API compatibility failed: {exc}. Reinstall "
                f"documented versions in a fresh environment."
            )
    return report
