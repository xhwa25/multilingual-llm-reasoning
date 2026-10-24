"""Offline integrity checks for the Phase 16K Colab wrapper; no CUDA or Drive access."""

import ast
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "phase16k_full_runner", ROOT / "scripts/phase16k_full_runner.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def fake_checkpoint(directory: Path, step: int) -> None:
    directory.mkdir(parents=True)
    for name in RUNNER.CHECKPOINT_FILES:
        (directory / name).write_bytes(b"fixture")
    (directory / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "max_steps": 75})
    )


def lightweight_hashes(directory: Path, step: int, *, load_state: bool = True) -> dict:
    assert directory.name == f"checkpoint-{step}"
    state = json.loads((directory / "trainer_state.json").read_text())
    assert state == {"global_step": step, "max_steps": 75}
    return {name: RUNNER.sha256(directory / name) for name in RUNNER.CHECKPOINT_FILES}


def test_actual_full_config_contract_and_frozen_data(private_final_run_inputs: None):
    contract, configs = RUNNER.paired_contract()
    assert contract["revision"] == "aa8e72537993ba99e69dfaafa59ed015b17504d1"
    assert contract["dataset_sha256"] == {
        "english_only_sft_final.jsonl": (
            "6e59b5344db55eb84f87c9dc9059587fbd00dae0edcea967a76d37705b8ed881"
        ),
        "multilingual_sft_final.jsonl": (
            "9b5c813da240229a31a7e57d1bc8a8e4cb9e8a410a837c38b01db23202df2021"
        ),
    }
    assert configs["english"].output_dir != configs["multilingual"].output_dir
    assert configs["english"].save_steps == configs["multilingual"].save_steps == 25
    assert configs["english"].batch_size * configs["english"].gradient_accumulation_steps == 8


def test_drive_root_refuses_occupied_or_changed_experiment(tmp_path):
    root = tmp_path / "phase16k"
    contract = {"revision": "same", "dataset_sha256": {"a": "b"}}
    RUNNER.ensure_drive_root(root, contract, resume=False)
    RUNNER.ensure_drive_root(root, contract, resume=False)
    with pytest.raises(ValueError, match="occupied|different"):
        RUNNER.ensure_drive_root(root, {"revision": "changed"}, resume=False)
    occupied = tmp_path / "other"
    occupied.mkdir()
    with pytest.raises(ValueError, match="occupied|different"):
        RUNNER.ensure_drive_root(occupied, contract, resume=False)


def test_environment_persistence_refuses_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(RUNNER, "ROOT", tmp_path)
    source = tmp_path / ".venv-colab/environment.freeze.txt"
    source.parent.mkdir()
    source.write_text("torch==2.8.0\n")
    drive = tmp_path / "drive"
    drive.mkdir()
    RUNNER.persist_environment(drive)
    RUNNER.persist_environment(drive)
    assert (drive / "environment.freeze.txt").read_bytes() == source.read_bytes()
    source.write_text("torch==changed\n")
    with pytest.raises(ValueError, match="differs"):
        RUNNER.persist_environment(drive)


def test_checkpoint_mirroring_is_verified_and_cannot_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(RUNNER, "checkpoint_hashes", lightweight_hashes)
    source = tmp_path / "local/checkpoint-25"
    destination = tmp_path / "drive/checkpoint-25"
    fake_checkpoint(source, 25)
    destination.parent.mkdir()
    RUNNER.mirror_checkpoint(source, destination, 25)
    assert RUNNER.mirrored_checkpoint_hashes(destination, 25) == lightweight_hashes(source, 25)
    RUNNER.mirror_checkpoint(source, destination, 25)
    (source / "optimizer.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        RUNNER.mirror_checkpoint(source, destination, 25)
    (destination / "scheduler.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        RUNNER.mirrored_checkpoint_hashes(destination, 25)


def test_restore_requires_matching_identity_and_ignores_completed_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(RUNNER, "checkpoint_hashes", lightweight_hashes)
    drive = tmp_path / "drive"
    local = tmp_path / "local"
    cp = drive / "checkpoints/checkpoint-50"
    fake_checkpoint(cp, 50)
    hashes = lightweight_hashes(cp, 50)
    (cp / "checkpoint_integrity.json").write_text(json.dumps({"global_step": 50, "sha256": hashes}))
    (drive / "run_at_checkpoint-50.json").write_text(json.dumps({"identity": {"a": 1}}))
    with pytest.raises(ValueError, match="identity"):
        RUNNER.restore_latest(drive, local, {"a": 2})
    assert RUNNER.restore_latest(drive, local, {"a": 1}) == 50
    assert (local / "checkpoint-50/optimizer.pt").is_file()
    assert not (local / "checkpoint-50/checkpoint_integrity.json").exists()
    with pytest.raises(ValueError, match="nonempty"):
        RUNNER.restore_latest(drive, local, {"a": 1})

    completed_drive = tmp_path / "completed-drive"
    completed = completed_drive / "checkpoints/checkpoint-75"
    fake_checkpoint(completed, 75)
    (completed / "checkpoint_integrity.json").write_text(
        json.dumps({"global_step": 75, "sha256": lightweight_hashes(completed, 75)})
    )
    with pytest.raises(ValueError, match="below step 75"):
        RUNNER.restore_latest(completed_drive, tmp_path / "new-local", {"a": 1})


def test_runner_requires_explicit_full_training_confirmation():
    source = (ROOT / "scripts/phase16k_full_runner.py").read_text()
    assert "--confirm-full-training" in source
    assert "--confirm-gpu-training" in source
    assert "--resume" in source
    assert 'mode="dry-run"' in source


def test_phase16k_notebook_safe_defaults_and_valid_code():
    notebook = json.loads((ROOT / "notebooks/phase16k_colab_full_training.ipynb").read_text())
    assert notebook["nbformat"] == 4
    assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])
    code = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and not cell["outputs"]
            source = "".join(cell["source"])
            ast.parse(source)
            code.append(source)
    joined = "\n".join(code)
    for switch in (
        "RUN_ENGLISH_FULL = False",
        "RUN_MULTILINGUAL_FULL = False",
        "PERSIST_FULL_TO_DRIVE = False",
    ):
        assert switch in joined
    assert "phase16k_full_runner.py" in joined
    assert "phase16i_cuda_preflight.py" in joined
    assert "--confirm-full-training" in joined
    assert "phase16j-smoke-v1" in joined
