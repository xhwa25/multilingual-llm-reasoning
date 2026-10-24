"""Training guardrails use synthetic data, fake hardware and no downloads or GPU work."""

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from test_sft import config as sft_source  # noqa: F401
from test_sft import tokenizer_loader

from multilingual_llm_reasoning import cli, training, training_hardware
from multilingual_llm_reasoning.sft import prepare_sft
from multilingual_llm_reasoning.training_config import config_dict, load_training_config
from multilingual_llm_reasoning.training_data import ResponseCollator, load_training_data
from multilingual_llm_reasoning.training_runtime import assert_adapter_only, assert_finite_losses

ROOT = Path(__file__).resolve().parents[1]


def test_finite_loss_guard():
    assert_finite_losses({"loss": 1.25, "eval_loss": 0.9, "learning_rate": 0.0002})
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="Non-finite"):
            assert_finite_losses({"loss": bad})


def test_phase16i_frozen_response_masks_and_padding(private_final_run_inputs: None):
    configs = [
        load_training_config(ROOT / "configs/phase16i_english_lora.yaml"),
        load_training_config(ROOT / "configs/phase16i_multilingual_lora.yaml"),
    ]
    for training_config in configs:
        rows, validation, manifest = load_training_data(training_config)
        assert len(rows) == 197 and len(validation) == 100
        assert manifest["approved_freeze_manifest"].endswith("dataset_freeze_manifest.json")
        for row in rows:
            boundary = row["prompt_tokens"]
            assert row["labels"][:boundary] == [-100] * boundary
            assert row["labels"][boundary:] == row["input_ids"][boundary:]
            assert any(value != -100 for value in row["labels"][boundary:])
        short, long = sorted(rows, key=lambda row: row["token_length"])[:: len(rows) - 1]
        padded = ResponseCollator(151645).pad([short, long])
        assert padded["labels"][0][short["token_length"] :] == [-100] * (
            long["token_length"] - short["token_length"]
        )
        assert padded["attention_mask"][0][short["token_length"] :] == [0] * (
            long["token_length"] - short["token_length"]
        )


def test_phase16i_matched_configs():
    for suffix in ("lora", "smoke"):
        a = config_dict(load_training_config(ROOT / f"configs/phase16i_english_{suffix}.yaml"))
        b = config_dict(load_training_config(ROOT / f"configs/phase16i_multilingual_{suffix}.yaml"))
        for item in (a, b):
            for key in ("name", "condition", "output_dir"):
                item.pop(key)
        assert a == b


def test_phase16i_freeze_authentication(tmp_path, private_final_run_inputs: None):
    config = load_training_config(ROOT / "configs/phase16i_english_lora.yaml")
    manifest = json.loads(config.sft_manifest.read_text())
    manifest["approved_freeze_sha256"] = "0" * 64
    manifest["approved_freeze_manifest"] = str(
        (config.sft_manifest.parent / manifest["approved_freeze_manifest"]).resolve()
    )
    tampered = tmp_path / "manifest.json"
    tampered.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="freeze manifest hash mismatch"):
        load_training_data(replace(config, sft_manifest=tampered))


@pytest.fixture
def config(sft_source, tmp_path):  # noqa: F811
    prepare_sft(sft_source, tokenizer_loader=tokenizer_loader)
    return replace(
        load_training_config(ROOT / "configs/english_lora.yaml"),
        sft_manifest=sft_source.output_dir / "manifest.json",
        output_dir=tmp_path / "training",
        revision="a" * 40,
        max_seq_length=2048,
    )


def test_pair_budgets_identical():
    a = config_dict(load_training_config(ROOT / "configs/english_lora.yaml"))
    b = config_dict(load_training_config(ROOT / "configs/multilingual_lora.yaml"))
    for key in ("name", "condition", "output_dir"):
        assert a.pop(key) != b.pop(key)
    assert a == b


@pytest.mark.parametrize(
    "key,value",
    [
        ("lora_rank", 0),
        ("qlora", "true"),
        ("precision", "fp32"),
        ("learning_rate", float("nan")),
        ("epochs", 0),
        ("target_modules", ["lm_head"]),
        ("revision", "main"),
        ("schema_version", True),
        ("batch_size", True),
    ],
)
def test_bad_config(tmp_path, key, value):
    raw = yaml.safe_load((ROOT / "configs/english_lora.yaml").read_text())
    raw[key] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_training_config(path)


def test_data_and_loss_mask(config):
    train, val, manifest = load_training_data(config)
    assert len(train) == 8 and len(val) == 4
    assert {r["original_problem_id"] for r in train}.isdisjoint(
        r["original_problem_id"] for r in val
    )
    # Padding with the EOS ID must mask padding, not actual assistant EOS tokens.
    batch = ResponseCollator(3).pad(
        [
            {"input_ids": [1, 2, 3], "labels": [-100, 2, 3], "attention_mask": [1, 1, 1]},
            {"input_ids": [1, 3], "labels": [-100, 3], "attention_mask": [1, 1]},
        ]
    )
    assert batch["labels"] == [[-100, 2, 3], [-100, 3, -100]]
    assert batch["attention_mask"] == [[1, 1, 1], [1, 1, 0]]
    with pytest.raises(ValueError, match="max_seq_length"):
        load_training_data(replace(config, max_seq_length=2))
    with pytest.raises(ValueError, match="revision"):
        load_training_data(replace(config, revision="b" * 40))
    manifest["training_ready"] = False
    config.sft_manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not training-ready"):
        load_training_data(config)


def test_tampered_hash_and_supervision(config):
    manifest = json.loads(config.sft_manifest.read_text())
    artifact = manifest["artifacts"]["english_train"]
    path = config.sft_manifest.parent / artifact["path"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["labels"] = rows[0]["input_ids"]
    content = "".join(json.dumps(row) + "\n" for row in rows)
    path.write_text(content)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_training_data(config)
    artifact["sha256"] = hashlib.sha256(content.encode()).hexdigest()
    config.sft_manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="loss mask"):
        load_training_data(config)


def test_confirmation_precedes_hardware(config, monkeypatch):
    monkeypatch.setattr(training, "probe_hardware", lambda c: pytest.fail("should not probe"))
    for mode in ("smoke", "full"):
        with pytest.raises(ValueError, match="explicit confirmation"):
            training.train_adapter(config, mode=mode)
    assert not config.output_dir.exists()


def test_dry_run_and_hardware_block(config, monkeypatch):
    monkeypatch.setattr(training, "probe_hardware", lambda c: {"blockers": ["No CUDA"]})
    result = training.train_adapter(config, check_hardware=True)
    assert result["optimizer_steps"] == 3
    assert result["blockers"] == ["No CUDA"]
    assert not result["training_started"] and not config.output_dir.exists()
    with pytest.raises(ValueError, match="before model download"):
        training.train_adapter(config, mode="smoke", confirm_gpu=True)


def test_runtime_dispatch_and_failure_record(config, monkeypatch):
    import multilingual_llm_reasoning.training_runtime as runtime

    monkeypatch.setattr(training, "probe_hardware", lambda c: {"blockers": []})
    seen = []

    def fake_run(cfg, mode, directory, checkpoint, train, val, manifest):
        seen.append((mode, checkpoint))
        assert len(train) == 8 and len(val) == 4
        return {"global_step": 20}

    monkeypatch.setattr(runtime, "execute_training", fake_run)
    report = training.train_adapter(config, mode="smoke", confirm_gpu=True)
    assert report["status"] == "completed" and seen == [("smoke", None)]
    assert report["preflight"]["optimizer_steps"] == 20
    assert not (config.output_dir / "full").exists()
    with pytest.raises(ValueError, match="nonempty"):
        training.train_adapter(config, mode="smoke", confirm_gpu=True)

    def fail(*args):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(runtime, "execute_training", fail)
    with pytest.raises(ValueError, match="no full-fine-tuning fallback"):
        training.train_adapter(config, mode="full", confirm_gpu=True)
    saved = json.loads((config.output_dir / "full/run.json").read_text())
    assert saved["status"] == "failed" and saved["wall_seconds_this_invocation"] >= 0


def test_resume_identity_and_state(config):
    directory = config.output_dir / "full"
    directory.mkdir(parents=True)
    identity = training.run_identity(config, "full")
    training.write_json(directory / "run.json", {"identity": identity})
    checkpoint = directory / "checkpoint-5"
    checkpoint.mkdir()
    for name in (
        "adapter_model.safetensors",
        "adapter_config.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "scaler.pt",
    ):
        (checkpoint / name).write_text("fixture")
    training.write_json(checkpoint / "trainer_state.json", {"global_step": 5, "max_steps": 75})
    assert training.resolve_resume(directory, "latest", identity) == checkpoint
    with pytest.raises(ValueError, match="mismatch"):
        training.resolve_resume(directory, "latest", {**identity, "mode": "smoke"})
    with pytest.raises(ValueError, match="must belong"):
        training.resolve_resume(directory, str(directory.parent / "checkpoint-5"), identity)
    (checkpoint / "optimizer.pt").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        training.resolve_resume(directory, "latest", identity)


def test_base_parameters_never_train():
    adapter = SimpleNamespace(requires_grad=True, numel=lambda: 100)
    base = SimpleNamespace(requires_grad=False, numel=lambda: 1000)
    model = SimpleNamespace(
        named_parameters=lambda: [("q.lora_A", adapter), ("q.weight", base)],
        parameters=lambda: [adapter, base],
    )
    assert assert_adapter_only(model)["trainable_parameters"] == 100
    base.requires_grad = True
    with pytest.raises(ValueError, match="only LoRA"):
        assert_adapter_only(model)


def test_cuda_vram_precision_and_packages(config, monkeypatch):
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        set_device=lambda i: None,
        mem_get_info=lambda i: (4 * 2**30, 16 * 2**30),
        get_device_capability=lambda i: (7, 5),
        get_device_name=lambda i: "T4",
        is_bf16_supported=lambda: False,
    )
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="12.6"))
    )
    monkeypatch.setattr(training_hardware, "version", lambda p: "0.0.0")
    report = training_hardware.probe_hardware(replace(config, precision="bf16"))
    assert report["free_vram_gib"] == 4
    assert any("Insufficient" in b for b in report["blockers"])
    assert any("BF16" in b for b in report["blockers"])
    assert any("unsupported" in b for b in report["blockers"])
    cuda.is_available = lambda: False
    assert any(
        "CUDA is required" in b for b in training_hardware.probe_hardware(config)["blockers"]
    )
    assert training_hardware.required_vram_gib(
        replace(config, qlora=False)
    ) > training_hardware.required_vram_gib(config)


def test_cli_dry_run_no_training(monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "train_adapter", lambda *a, **kw: {"ready": False, "blockers": ["GPU unavailable"]}
    )
    with pytest.raises(SystemExit) as exc:
        cli.main(["train", "--config", str(ROOT / "configs/english_lora.yaml")])
    assert exc.value.code == 2 and "GPU unavailable" in capsys.readouterr().out


def test_runtime_20_step_contract_with_fake_stack(config, monkeypatch):
    """Exercise runtime wiring and adapter serialization, not numerical GPU convergence."""
    from test_sft import FakeTokenizer

    import multilingual_llm_reasoning.training_runtime as runtime
    from multilingual_llm_reasoning.training_runtime import execute_training

    train, val, manifest = load_training_data(config)
    tokenizer = FakeTokenizer()
    tokenizer.pad_token_id = 0
    saved = []
    digests = iter(("before_optimizer", "after_optimizer"))
    monkeypatch.setattr(runtime, "trainable_digest", lambda model: next(digests))
    monkeypatch.setattr(
        runtime,
        "verify_smoke_checkpoint",
        lambda directory, precision: saved.append(("checkpoint_loaded", directory, precision)),
    )
    tokenizer.save_pretrained = lambda path: saved.append(("tokenizer", Path(path)))
    manifest["tokenizer"]["chat_template_sha256"] = hashlib.sha256(
        tokenizer.get_chat_template().encode()
    ).hexdigest()
    param = SimpleNamespace(
        requires_grad=True, numel=lambda: 10, device=SimpleNamespace(type="cuda")
    )
    model = SimpleNamespace(
        config=SimpleNamespace(use_cache=True),
        named_modules=lambda: [(name, None) for name in config.target_modules],
        named_parameters=lambda: [("q.lora_A.weight", param)],
        parameters=lambda: [param],
        save_pretrained=lambda path, **kw: saved.append(("adapter", Path(path), kw)),
    )
    cuda = SimpleNamespace(
        set_device=lambda i: None,
        reset_peak_memory_stats=lambda i: None,
        memory_allocated=lambda: 100,
        memory_reserved=lambda: 200,
        max_memory_allocated=lambda: 150,
        max_memory_reserved=lambda: 250,
        synchronize=lambda: None,
    )
    fake_torch = SimpleNamespace(cuda=cuda, bfloat16="bf16", float16="fp16")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(Dataset=SimpleNamespace(from_list=lambda rows: rows)),
    )
    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(
            LoraConfig=lambda **kw: kw,
            get_peft_model=lambda model, cfg: model,
            prepare_model_for_kbit_training=lambda *a, **kw: pytest.fail("QLoRA not selected"),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **kw: model),
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer),
            BitsAndBytesConfig=lambda **kw: kw,
            TrainerCallback=object,
            set_seed=lambda seed: None,
        ),
    )
    observed = {}

    class Args:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            observed.update(kwargs)

        def to_dict(self):
            return self.__dict__

    class Trainer:
        def __init__(self, **kwargs):
            self.model, self.args = kwargs["model"], kwargs["args"]
            self.state = SimpleNamespace(global_step=0, epoch=0)
            self.callback = kwargs["callbacks"][0]
            assert kwargs["train_dataset"][0]["labels"] == train[0]["labels"]
            assert isinstance(kwargs["data_collator"], ResponseCollator)

        def evaluate(self, metric_key_prefix="eval"):
            return {metric_key_prefix + "_loss": 1.0}

        def train(self, resume_from_checkpoint=None):
            assert resume_from_checkpoint is None
            self.state.global_step = 20
            self.callback.on_log(self.args, self.state, None, {"loss": 1.0})
            return SimpleNamespace(metrics={"train_loss": 1.0})

        def save_model(self, path):
            self._save(path)

        def save_state(self):
            pass

    monkeypatch.setitem(sys.modules, "trl", SimpleNamespace(SFTConfig=Args, SFTTrainer=Trainer))
    directory = config.output_dir / "smoke"
    directory.mkdir(parents=True)
    metrics = execute_training(
        replace(config, qlora=False), "smoke", directory, None, train, val, manifest
    )
    assert metrics["global_step"] == 20 and metrics["eval_loss"] == 1
    assert metrics["smoke_optimizer_update_verified"]
    assert metrics["smoke_checkpoint_load_verified"]
    assert any(item[0] == "checkpoint_loaded" for item in saved)
    assert observed["max_steps"] == 20
    assert observed["dataset_kwargs"] == {"skip_prepare_dataset": True}
    assert observed["packing"] is False and observed["save_only_model"] is False
    assert observed["report_to"] == "none" and observed["push_to_hub"] is False
    adapter = next(row for row in saved if row[0] == "adapter")
    assert adapter[2] == {"safe_serialization": True, "save_embedding_layers": False}
    assert (directory / "training_log.jsonl").is_file()
    assert (directory / "adapter/training_args.json").is_file()
