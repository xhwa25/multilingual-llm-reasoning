"""Guarded Colab full-run launcher with verified, in-flight Drive checkpoint mirroring.

This wraps the existing trainer; it does not implement model training.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from multilingual_llm_reasoning.training import run_identity, train_adapter
from multilingual_llm_reasoning.training_config import load_training_config

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "outputs/phase16h-approved-sft-freeze-v1"
CONFIGS = {
    "english": ROOT / "configs/phase16i_english_lora.yaml",
    "multilingual": ROOT / "configs/phase16i_multilingual_lora.yaml",
}
CHECKPOINT_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "scaler.pt",
)
EXPECTED_STEPS = (25, 50, 75)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_hashes(directory: Path, step: int, *, load_state: bool = True) -> dict[str, str]:
    """Admit only a complete, deserializable checkpoint at the expected optimizer step."""
    if directory.name != f"checkpoint-{step}" or step not in EXPECTED_STEPS:
        raise ValueError("Unexpected checkpoint name or optimizer step")
    for name in CHECKPOINT_FILES:
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Incomplete checkpoint-{step}: {name}")
    state = json.loads((directory / "trainer_state.json").read_text())
    if state.get("global_step") != step or state.get("max_steps") != 75:
        raise ValueError("Checkpoint trainer state does not match the 75-step budget")
    if load_state:
        import torch
        from safetensors import safe_open

        with safe_open(directory / "adapter_model.safetensors", framework="pt") as handle:
            if not list(handle.keys()):
                raise ValueError("Empty LoRA adapter checkpoint")
        optimizer = torch.load(directory / "optimizer.pt", map_location="cpu", weights_only=True)
        scheduler = torch.load(directory / "scheduler.pt", map_location="cpu", weights_only=True)
        if not optimizer.get("state") or not scheduler:
            raise ValueError("Optimizer or scheduler state is empty")
        torch.load(directory / "scaler.pt", map_location="cpu", weights_only=True)
    return {name: sha256(directory / name) for name in CHECKPOINT_FILES}


def mirrored_checkpoint_hashes(directory: Path, step: int) -> dict[str, str]:
    manifest = json.loads((directory / "checkpoint_integrity.json").read_text())
    hashes = checkpoint_hashes(directory, step)
    if manifest != {"global_step": step, "sha256": hashes}:
        raise ValueError(f"Drive checkpoint-{step} failed integrity validation")
    return hashes


def mirror_checkpoint(source: Path, destination: Path, step: int) -> None:
    """Copy to a unique staging directory; publish only after hash verification."""
    source_hashes = checkpoint_hashes(source, step)
    if destination.exists():
        if mirrored_checkpoint_hashes(destination, step) != source_hashes:
            raise ValueError(f"Existing Drive checkpoint differs: {destination}")
        return
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.partial")
    shutil.copytree(source, temporary)
    try:
        for name, digest in source_hashes.items():
            if sha256(temporary / name) != digest or sha256(source / name) != digest:
                raise ValueError(f"Checkpoint changed during copy: {source}")
        (temporary / "checkpoint_integrity.json").write_text(
            json.dumps({"global_step": step, "sha256": source_hashes}, indent=2) + "\n"
        )
        if destination.exists():
            raise FileExistsError(f"Checkpoint destination appeared during copy: {destination}")
        temporary.rename(destination)
        mirrored_checkpoint_hashes(destination, step)
    except Exception:
        # Preserve the staged copy for diagnosis rather than silently deleting evidence.
        raise


def paired_contract() -> tuple[dict, dict]:
    left, right = [load_training_config(path) for path in CONFIGS.values()]
    a, b = asdict(left), asdict(right)
    for item in (a, b):
        for key in ("name", "condition", "output_dir"):
            item.pop(key)
    if a != b:
        raise ValueError("Full-training configurations are not matched")
    if left.output_dir == right.output_dir:
        raise ValueError("Full-training output directories must be distinct")
    if any(c.revision != "aa8e72537993ba99e69dfaafa59ed015b17504d1" for c in (left, right)):
        raise ValueError("Pinned model revision changed")
    if any(c.save_steps != 25 or c.save_total_limit != 2 for c in (left, right)):
        raise ValueError("Expected 25-step checkpoint schedule changed")
    freeze = json.loads((FREEZE / "dataset_freeze_manifest.json").read_text())
    if not freeze.get("training_ready") or freeze["tokenizer_revision"] != left.revision:
        raise ValueError("Approved freeze or tokenizer revision differs")
    hashes = {}
    for name in ("english_only_sft_final.jsonl", "multilingual_sft_final.jsonl"):
        actual = sha256(FREEZE / name)
        if actual != freeze["dataset_files_sha256"][name]:
            raise ValueError(f"Frozen dataset hash mismatch: {name}")
        hashes[name] = actual
    for config in (left, right):
        report = train_adapter(config, mode="dry-run")
        if not report["ready"] or report["optimizer_steps"] != 75:
            raise ValueError("Full training dry-run failed or optimizer budget is not 75")
        if report["train_examples"] != 197 or not report["response_only_loss"]:
            raise ValueError("Training examples or response-only loss changed")
    contract = {
        "model": left.model,
        "revision": left.revision,
        "dataset_sha256": hashes,
        "config_sha256": {name: sha256(path) for name, path in CONFIGS.items()},
    }
    return contract, {"english": left, "multilingual": right}


def ensure_drive_root(drive_root: Path, contract: dict, *, resume: bool) -> None:
    marker = drive_root / "experiment_contract.json"
    if not drive_root.exists():
        if resume:
            raise FileNotFoundError("Drive experiment root does not exist for resume")
        drive_root.mkdir(parents=True)
        marker.write_text(json.dumps(contract, indent=2) + "\n")
    elif not marker.is_file() or json.loads(marker.read_text()) != contract:
        raise ValueError("Drive destination is occupied or has a different experiment contract")


def persist_environment(drive_root: Path) -> None:
    source = ROOT / ".venv-colab/environment.freeze.txt"
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError("Locked Colab environment freeze is missing")
    destination = drive_root / "environment.freeze.txt"
    if destination.exists():
        if sha256(destination) != sha256(source):
            raise ValueError("Drive environment differs from this Colab installation")
    else:
        shutil.copy2(source, destination)
        if sha256(destination) != sha256(source):
            raise ValueError("Drive environment copy failed hash verification")


def restore_latest(drive_condition: Path, local_run: Path, identity: dict) -> int:
    if local_run.exists() and any(local_run.iterdir()):
        raise ValueError("Local full-run directory is nonempty; refuse to overwrite it")
    candidates = []
    for step in EXPECTED_STEPS:
        checkpoint = drive_condition / "checkpoints" / f"checkpoint-{step}"
        if checkpoint.is_dir() and step < 75:
            mirrored_checkpoint_hashes(checkpoint, step)
            saved_run = drive_condition / f"run_at_checkpoint-{step}.json"
            if not saved_run.is_file() or json.loads(saved_run.read_text())["identity"] != identity:
                raise ValueError("Drive run identity differs from the frozen full-training run")
            candidates.append((step, checkpoint, saved_run))
    if not candidates:
        raise ValueError("No complete Drive checkpoint below step 75 is available to resume")
    step, checkpoint, saved_run = max(candidates)
    local_run.mkdir(parents=True, exist_ok=True)
    shutil.copytree(checkpoint, local_run / checkpoint.name)
    (local_run / checkpoint.name / "checkpoint_integrity.json").unlink()
    shutil.copy2(saved_run, local_run / "run.json")
    checkpoint_hashes(local_run / checkpoint.name, step)
    return step


def mirror_available(local_run: Path, drive_condition: Path, mirrored: set[int]) -> None:
    for step in EXPECTED_STEPS:
        source = local_run / f"checkpoint-{step}"
        if step in mirrored or not source.is_dir():
            continue
        try:
            hashes = checkpoint_hashes(source, step)
        except Exception:
            # The trainer may still be writing. Final verification rejects a
            # checkpoint that never becomes complete and deserializable.
            continue
        destination = drive_condition / "checkpoints" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        mirror_checkpoint(source, destination, step)
        if mirrored_checkpoint_hashes(destination, step) != hashes:
            raise ValueError("Drive checkpoint hash differs from the local checkpoint")
        run = local_run / "run.json"
        saved_run = drive_condition / f"run_at_checkpoint-{step}.json"
        if not run.is_file() or saved_run.exists():
            if not saved_run.exists():
                raise ValueError("Checkpoint has no run identity metadata")
        else:
            shutil.copy2(run, saved_run)
        mirrored.add(step)
        print(f"Persisted complete checkpoint-{step} to Drive", flush=True)


def verify_completed(local_run: Path) -> dict:
    run = json.loads((local_run / "run.json").read_text())
    metrics = json.loads((local_run / "metrics.json").read_text())
    state = json.loads((local_run / "trainer_state.json").read_text())
    if run.get("status") != "completed" or metrics.get("global_step") != 75:
        raise ValueError("Full run did not complete exactly 75 optimizer steps")
    if state.get("global_step") != 75 or state.get("max_steps") != 75:
        raise ValueError("Final trainer state differs from the 75-step budget")
    if not math.isfinite(float(metrics["train_loss"])) or not math.isfinite(
        float(metrics["eval_loss"])
    ):
        raise ValueError("Non-finite final training or validation loss")
    adapter = local_run / "adapter"
    for name in ("adapter_model.safetensors", "adapter_config.json", "evaluation_provenance.json"):
        if not (adapter / name).is_file():
            raise ValueError(f"Final LoRA adapter is missing {name}")
    checkpoint_hashes(local_run / "checkpoint-75", 75)
    return metrics


def verify_final_adapter(local_run: Path, config) -> dict:
    """Load the saved adapter onto a fresh pinned 4-bit base and run a tiny forward pass."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from multilingual_llm_reasoning.comparison_backends import activate_adapter

    adapter_path = local_run / "adapter"
    provenance = json.loads((adapter_path / "evaluation_provenance.json").read_text())
    if (
        provenance["model"] != config.model
        or provenance["revision"] != config.revision
        or provenance["condition"] != config.condition
        or provenance["training_budget"]["global_step"] != 75
    ):
        raise ValueError("Final adapter provenance differs from the full-training run")
    tokenizer = AutoTokenizer.from_pretrained(config.model, revision=config.revision)
    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        revision=config.revision,
        trust_remote_code=False,
        torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        ),
        device_map={"": config.device_index},
        attn_implementation="sdpa",
    )
    condition_id = "english_lora" if config.condition == "english_train" else "multilingual_lora"
    model, activation = activate_adapter(model, adapter_path, condition_id)
    tokens = tokenizer("Adapter load check.", return_tensors="pt").to(f"cuda:{config.device_index}")
    with torch.inference_mode():
        output = model(**tokens)
    if not torch.isfinite(output.logits).all():
        raise ValueError("Adapter forward pass produced non-finite logits")
    report = {
        "status": "passed",
        "model": config.model,
        "revision": config.revision,
        "condition": config.condition,
        "active_adapters": activation["active_adapters"],
        "forward_tokens": int(tokens["input_ids"].numel()),
    }
    (local_run / "adapter_load_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    del output, model
    torch.cuda.empty_cache()
    return report


def copy_final(local_run: Path, drive_condition: Path, metrics: dict, contract: dict) -> None:
    final = drive_condition / "final"
    if final.exists():
        raise FileExistsError(f"Final Drive outputs already exist: {final}")
    temporary = drive_condition / f".final.{uuid4().hex}.partial"
    temporary.mkdir()
    for name in (
        "run.json",
        "metrics.json",
        "training_log.jsonl",
        "trainer_state.json",
        "adapter_load_verification.json",
    ):
        source = local_run / name
        if not source.is_file() or source.stat().st_size == 0:
            raise ValueError(f"Required final training artifact is missing: {source}")
        shutil.copy2(source, temporary / name)
    shutil.copytree(local_run / "adapter", temporary / "adapter")
    hashes = {
        str(path.relative_to(temporary)): sha256(path)
        for path in temporary.rglob("*")
        if path.is_file()
    }
    (temporary / "completion.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "global_step": 75,
                "metrics": metrics,
                "contract": contract,
                "artifact_sha256": hashes,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    temporary.rename(final)
    for rel, digest in hashes.items():
        if sha256(final / rel) != digest:
            raise ValueError(f"Final Drive artifact failed verification: {rel}")


def run(condition: str, drive_root: Path, *, resume: bool, poll_seconds: float) -> None:
    if str(drive_root).startswith("/content/drive/") and not Path("/content/drive").is_mount():
        raise ValueError("Google Drive is not mounted; refusing local-only checkpoint copies")
    contract, configs = paired_contract()
    config = configs[condition]
    identity = run_identity(config, "full")
    ensure_drive_root(drive_root, contract, resume=resume)
    persist_environment(drive_root)
    drive_condition = drive_root / condition
    local_run = config.output_dir / "full"
    if resume:
        if not drive_condition.is_dir() or (drive_condition / "final").exists():
            raise ValueError("Cannot resume a missing or already-completed condition")
        resumed_step = restore_latest(drive_condition, local_run, identity)
        print(f"Restored verified checkpoint-{resumed_step}; budget remains 75 total steps")
    else:
        if drive_condition.exists() or (local_run.exists() and any(local_run.iterdir())):
            raise FileExistsError("Condition already has local or Drive outputs; refuse restart")
        drive_condition.mkdir()
        shutil.copy2(CONFIGS[condition], drive_condition / "training_config.yaml")
    args = [
        sys.executable,
        "-m",
        "multilingual_llm_reasoning",
        "train",
        "--config",
        str(CONFIGS[condition]),
        "--mode",
        "full",
        "--confirm-gpu-training",
    ]
    if resume:
        args.extend(("--resume", "latest"))
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    process = subprocess.Popen(args, cwd=ROOT, env=environment)
    mirrored = {
        step
        for step in EXPECTED_STEPS
        if (drive_condition / "checkpoints" / f"checkpoint-{step}").is_dir()
    }
    try:
        while process.poll() is None:
            mirror_available(local_run, drive_condition, mirrored)
            time.sleep(poll_seconds)
        mirror_available(local_run, drive_condition, mirrored)
    except Exception:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
        raise
    if process.returncode != 0:
        raise RuntimeError(
            f"Existing trainer exited {process.returncode}; inspect {local_run}/run.json"
        )
    metrics = verify_completed(local_run)
    if 75 not in mirrored:
        raise ValueError("Final checkpoint-75 was not persisted to Drive")
    if not any(step in mirrored for step in (25, 50)):
        raise ValueError("No intermediate checkpoint was persisted to Drive during training")
    for name, digest in contract["dataset_sha256"].items():
        if sha256(FREEZE / name) != digest:
            raise ValueError("Frozen dataset changed during training")
    verify_final_adapter(local_run, config)
    copy_final(local_run, drive_condition, metrics, contract)
    print(
        json.dumps(
            {
                "condition": condition,
                "status": "completed",
                "global_step": 75,
                "drive": str(drive_condition),
                "runtime_seconds": metrics["runtime_seconds"],
                "peak_reserved_gib": metrics["peak_reserved_gib"],
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONFIGS, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--resume-from-drive", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--confirm-full-training", action="store_true")
    args = parser.parse_args()
    if not args.confirm_full_training:
        raise SystemExit("Full training requires --confirm-full-training")
    if args.poll_seconds <= 0 or not math.isfinite(args.poll_seconds):
        raise SystemExit("--poll-seconds must be positive and finite")
    run(
        args.condition,
        args.drive_root.resolve(),
        resume=args.resume_from_drive,
        poll_seconds=args.poll_seconds,
    )


if __name__ == "__main__":
    main()
