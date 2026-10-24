"""Read-only paired CUDA admission report for approved Phase 16H datasets."""

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from multilingual_llm_reasoning.training import train_adapter
from multilingual_llm_reasoning.training_config import load_training_config

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "outputs/phase16h-approved-sft-freeze-v1"
DEFAULT_OUTPUT = ROOT / "outputs/phase16i-training-preparation-v1/cuda_preflight.json"
MIN_FREE_DISK_GIB = 20.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = [
        ROOT / "configs/phase16i_english_lora.yaml",
        ROOT / "configs/phase16i_multilingual_lora.yaml",
    ]
    configs = [load_training_config(path) for path in paths]
    left, right = (asdict(config) for config in configs)
    for settings in (left, right):
        for key in ("name", "condition", "output_dir"):
            settings.pop(key)
    if left != right:
        raise ValueError("Training conditions have unmatched hyperparameters or datasets")
    manifest = json.loads((FREEZE / "dataset_freeze_manifest.json").read_text())
    frozen_hashes = {}
    for name in ("english_only_sft_final.jsonl", "multilingual_sft_final.jsonl"):
        actual = hashlib.sha256((FREEZE / name).read_bytes()).hexdigest()
        if actual != manifest["dataset_files_sha256"][name]:
            raise ValueError(f"Frozen dataset SHA-256 mismatch: {name}")
        frozen_hashes[name] = actual
    if manifest.get("training_ready") is not True:
        raise ValueError("Frozen dataset is not approved for training")
    free_disk_gib = shutil.disk_usage(ROOT).free / 2**30
    results = [train_adapter(config, mode="dry-run", check_hardware=True) for config in configs]
    blockers = sorted({item for result in results for item in result["blockers"]})
    qlora_kernel_probe = "not_run"
    if not blockers and configs[0].qlora:
        try:
            import bitsandbytes as bnb
            import torch

            dtype = torch.bfloat16 if configs[0].precision == "bf16" else torch.float16
            probe = torch.ones((64, 64), device="cuda:0", dtype=dtype)
            bnb.functional.quantize_4bit(probe, quant_type="nf4")
            del probe
            qlora_kernel_probe = "passed"
        except Exception as exc:
            qlora_kernel_probe = "failed"
            blockers.append(f"bitsandbytes CUDA NF4 kernel probe failed: {exc}")
    if free_disk_gib < MIN_FREE_DISK_GIB:
        blockers.append(
            f"Checkpoint disk headroom below {MIN_FREE_DISK_GIB:g} GiB; free space before training."
        )
    report = {
        "status": "ready" if not blockers else "blocked",
        "training_started": False,
        "frozen_dataset_hashes": frozen_hashes,
        "model": configs[0].model,
        "model_revision": configs[0].revision,
        "tokenizer_revision": manifest["tokenizer_revision"],
        "matched_settings": True,
        "qlora_kernel_probe": qlora_kernel_probe,
        "disk_free_gib": free_disk_gib,
        "minimum_disk_headroom_gib": MIN_FREE_DISK_GIB,
        "conditions": {
            "english": results[0],
            "multilingual": results[1],
        },
        "blockers": blockers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "disk_free_gib": free_disk_gib,
                "blockers": blockers,
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
