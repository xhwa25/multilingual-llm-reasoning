"""Local/Colab boundary, real-data export states and notebook validation without downloads."""

import ast
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_sft import config as sft_source  # noqa: F401
from test_sft import tokenizer_loader

from multilingual_llm_reasoning import training
from multilingual_llm_reasoning.sft import prepare_sft
from multilingual_llm_reasoning.sft_completion import complete_sft
from multilingual_llm_reasoning.training_config import load_training_config
from multilingual_llm_reasoning.training_data import load_training_data

ROOT = Path(__file__).resolve().parents[1]


def train_config(sft):
    return replace(
        load_training_config(ROOT / "configs/english_lora.yaml"),
        sft_manifest=sft.output_dir / "manifest.json",
        revision="a" * 40,
        max_seq_length=2048,
    )


def test_partial_real_english_no_translation_required(sft_source):  # noqa: F811
    source = replace(sft_source, translation_manifest=sft_source.output_dir / "absent.json")
    report = prepare_sft(source, english_only=True, tokenizer_loader=tokenizer_loader)
    assert report["status"] == "partial" and not report["training_ready"]
    assert report["conditions"]["english_train"]["training_ready"]
    assert report["conditions"]["multilingual_train"]["status"] == "missing"
    assert "multilingual_train" not in report["artifacts"]
    assert report["audit"]["matched_original_ids"] is None
    cfg = train_config(source)
    rows, validation, _ = load_training_data(cfg)
    assert len(rows) == 8 and len(validation) == 4
    assert all(row["question_text"] == row["original_question"] for row in rows)
    with pytest.raises(ValueError, match="Multilingual SFT is missing"):
        load_training_data(replace(cfg, condition="multilingual_train"))


def test_complete_keeps_english_and_validation(sft_source):  # noqa: F811
    partial = prepare_sft(sft_source, english_only=True, tokenizer_loader=tokenizer_loader)
    full = complete_sft(sft_source, tokenizer_loader=tokenizer_loader)
    assert full["training_ready"] and full["status"] == "ready"
    for name in ("english_train", "validation"):
        assert partial["artifacts"][name] == full["artifacts"][name]
    a, val, _ = load_training_data(train_config(sft_source))
    b, _, _ = load_training_data(replace(train_config(sft_source), condition="multilingual_train"))
    assert {r["original_problem_id"] for r in a} == {r["original_problem_id"] for r in b}
    assert not {r["original_problem_id"] for r in a} & {r["original_problem_id"] for r in val}


def test_failed_completion_preserves_partial(sft_source):  # noqa: F811
    prepare_sft(sft_source, english_only=True, tokenizer_loader=tokenizer_loader)
    manifest = sft_source.output_dir / "manifest.json"
    before = manifest.read_bytes()
    changed = replace(sft_source, seed=sft_source.seed + 1)
    with pytest.raises(ValueError, match="changed English/validation"):
        complete_sft(changed, tokenizer_loader=tokenizer_loader)
    assert manifest.read_bytes() == before
    assert not (sft_source.output_dir / "multilingual_train.jsonl").exists()


def test_local_dry_run_never_probes_gpu(sft_source, monkeypatch):  # noqa: F811
    prepare_sft(sft_source, english_only=True, tokenizer_loader=tokenizer_loader)
    monkeypatch.setattr(
        training, "probe_hardware", lambda c: pytest.fail("GPU probe in local mode")
    )
    report = training.train_adapter(train_config(sft_source))
    assert report["ready"] and report["validation_ready"]
    assert report["gpu_ready"] is None and not report["training_ready"]
    report = training.train_adapter(
        replace(train_config(sft_source), condition="multilingual_train")
    )
    assert report["data_status"] == "missing" and not report["ready"]


def test_resume_identity_portable(sft_source, tmp_path):  # noqa: F811
    prepare_sft(sft_source, english_only=True, tokenizer_loader=tokenizer_loader)
    cfg = train_config(sft_source)
    other = tmp_path / "relocated.json"
    other.write_bytes(cfg.sft_manifest.read_bytes())
    assert training.run_identity(cfg, "full") == training.run_identity(
        replace(cfg, sft_manifest=other, output_dir=tmp_path / "colab-output"), "full"
    )


def test_dependency_lock_and_setup():
    import tomllib

    lock = (ROOT / "requirements/colab.lock").read_text()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    for name, version in {
        "torch": "2.8.0+cu126",
        "transformers": "4.56.2",
        "peft": "0.17.1",
        "trl": "0.23.1",
        "accelerate": "1.10.1",
        "datasets": "4.1.1",
        "bitsandbytes": "0.47.0",
    }.items():
        assert f"{name}=={version}" in lock
    assert "torch==2.8.0" in project["project"]["optional-dependencies"]["train"]
    setup = (ROOT / "scripts/setup_colab.sh").read_text()
    assert "--require-hashes" in setup and "--extra-index-url" in setup
    assert "3.11.14" in setup and "--no-build-isolation" in setup
    # Every locked distribution has a hash, rather than merely top-level pins.
    for block in lock.split("\n"):
        if block and block[0].isalnum() and "==" in block:
            start = lock.index(block)
            assert "--hash=sha256:" in lock[start:].split("\n", 2)[1]


def test_artifact_corruption_blocks_completion(sft_source):  # noqa: F811
    report = prepare_sft(sft_source, english_only=True, tokenizer_loader=tokenizer_loader)
    path = sft_source.output_dir / report["artifacts"]["english_train"]["path"]
    path.write_text(path.read_text() + "\n")
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        != report["artifacts"]["english_train"]["sha256"]
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        complete_sft(sft_source, tokenizer_loader=tokenizer_loader)


def test_bundle_excludes_secrets_weights_and_preview(tmp_path, monkeypatch):
    import importlib.util
    import tarfile

    spec = importlib.util.spec_from_file_location("bundle_script", ROOT / "scripts/bundle_colab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    for name in (
        "src/code.py",
        "src/.env",
        "src/model.safetensors",
        "src/__pycache__/x.pyc",
        "data/sft/gsm8k-200/manifest.json",
        "data/sft/gsm8k-200-preview/manifest.json",
        "docs/licenses/GSM8K_LICENSE.txt",
        ".env",
    ):
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("fixture")
    output = tmp_path / "bundle.tar.gz"
    module.bundle(output)
    with tarfile.open(output) as archive:
        names = archive.getnames()
    assert any(name.endswith("gsm8k-200/manifest.json") for name in names)
    assert any(name.endswith("GSM8K_LICENSE.txt") for name in names)
    assert not any(
        ".env" in name or "preview" in name or "safetensors" in name or "pycache" in name
        for name in names
    )


def test_phase16i_bundle_requires_and_includes_only_frozen_inputs(tmp_path, monkeypatch):
    import importlib.util
    import tarfile

    spec = importlib.util.spec_from_file_location("bundle_script", ROOT / "scripts/bundle_colab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    output = tmp_path / "phase16i.tar.gz"
    with pytest.raises(FileNotFoundError, match="incomplete"):
        module.bundle(output, phase16i_frozen=True)
    for name in module.PHASE16I_FROZEN_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    (tmp_path / "outputs/private.txt").write_text("secret")
    module.bundle(output, phase16i_frozen=True)
    with tarfile.open(output) as archive:
        names = set(archive.getnames())
    assert all(
        f"multilingual-llm-reasoning/{name}" in names for name in module.PHASE16I_FROZEN_FILES
    )
    assert "multilingual-llm-reasoning/outputs/private.txt" not in names


def test_phase16j_notebook_uses_explicit_smoke_switches_and_frozen_inputs():
    notebook = json.loads((ROOT / "notebooks/phase16j_colab_training.ipynb").read_text())
    assert notebook["nbformat"] == 4
    assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])
    sources = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None and not cell["outputs"]
        source = "".join(cell["source"])
        ast.parse(source)
        sources.append(source)
    joined = "\n".join(sources)
    assert "RUN_ENGLISH_SMOKE = False" in joined
    assert "RUN_MULTILINGUAL_SMOKE = False" in joined
    assert "PERSIST_TO_DRIVE = False" in joined
    assert '"--mode"' in joined and '"smoke"' in joined
    assert "--confirm-gpu-training" in joined
    assert "phase16i_cuda_preflight.py" in joined
    assert "phase16i_english_smoke.yaml" in joined
    assert "phase16i_multilingual_smoke.yaml" in joined
    assert "6e59b5344db55eb84f87c9dc9059587f" in joined
    assert "bd00dae0edcea967a76d37705b8ed881" in joined
    assert "9b5c813da240229a31a7e57d1bc8a8e" in joined
    assert "4cb9e8a410a837c38b01db23202df2021" in joined
