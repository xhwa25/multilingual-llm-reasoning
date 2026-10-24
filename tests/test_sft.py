"""Synthetic source/translation fixtures and a deterministic fake tokenizer; no downloads."""

import copy
import hashlib
import json
import socket
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from multilingual_llm_reasoning import cli, preparation
from multilingual_llm_reasoning.preparation import PreparationConfig, prepare_data
from multilingual_llm_reasoning.sft import prepare_sft, validate_sft_pairs
from multilingual_llm_reasoning.sft_config import SFTConfig, load_sft_config
from multilingual_llm_reasoning.sft_tokenization import tokenize_sft

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        pytest.fail("SFT tests must not download data or tokenizers")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


class FakeTokenizer:
    eos_token_id = ord("§")

    def __init__(self) -> None:
        self.calls = []

    def get_chat_template(self) -> str:
        return "synthetic-test-template"

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        self.calls.append((messages, kwargs))
        text = "".join(f"[{m['role']}]\n{m['content']}§\n" for m in messages)
        if kwargs["add_generation_prompt"]:
            text += "[assistant]\n"
        return [ord(char) for char in text] if kwargs["tokenize"] else text

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        assert kwargs == {"add_special_tokens": False, "truncation": False}
        return {"input_ids": [ord(char) for char in text]}


def tokenizer_loader(config: SFTConfig) -> tuple[FakeTokenizer, dict]:
    return FakeTokenizer(), {
        "checkpoint": config.tokenizer_checkpoint,
        "resolved_revision": "a" * 40,
        "chat_template": "synthetic-test-template",
    }


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SFTConfig:
    source = {
        split: [
            {
                "question": f"Question {split} {i}: add 3.5 and 9.",
                "answer": "Add: 3.5 + 9 = <<3.5+9=12.5>>12.5.\n#### 12.50",
            }
            for i in range(32 if split == "train" else 8)
        ]
        for split in ("train", "test")
    }
    monkeypatch.setattr(preparation, "load_source", lambda revision: (source, "a" * 40))
    prep_path = prepare_data(PreparationConfig(42, 8, 16, 4, "a" * 40, tmp_path / "prepared"))
    prep = json.loads(prep_path.read_text())
    train = rows(prep_path.parent / prep["subsets"]["train"]["path"])
    variants = []
    for original in train:
        for language in ("en", "zh", "de", "ja"):
            variants.append(
                {
                    "original_problem_id": original["original_problem_id"],
                    "language": language,
                    "source_text": original["original_question"],
                    "source_split": "train",
                    "split": "train",
                    "gold_answer": original["gold_answer"],
                    "translated_text": original["original_question"]
                    if language == "en"
                    else f"{language} fixture: {original['original_question']}",
                    "qa_status": "source_original" if language == "en" else "human_approved",
                    "translation_provider": "fixture",
                    "translation_model": "fixture-v1",
                }
            )
    translated = tmp_path / "translated"
    translated.mkdir()
    content = "".join(json.dumps(row) + "\n" for row in variants)
    (translated / "translations.jsonl").write_text(content)
    manifest = {
        "schema_version": 1,
        "complete": True,
        "mock": False,
        "input_manifest_sha256": hashlib.sha256(prep_path.read_bytes()).hexdigest(),
        "artifacts": {
            "variants": {
                "path": "translations.jsonl",
                "count": len(variants),
                "sha256": hashlib.sha256(content.encode()).hexdigest(),
            }
        },
    }
    (translated / "manifest.json").write_text(json.dumps(manifest))
    raw = yaml.safe_load((ROOT / "configs/prepare_sft.yaml").read_text())
    raw.update(
        preparation_manifest=str(prep_path),
        translation_manifest=str(translated / "manifest.json"),
        output_dir=str(tmp_path / "sft"),
        train_size=8,
        validation_size=4,
    )
    path = tmp_path / "sft.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_sft_config(path)


def test_matched_balanced_targets_masks_and_audit(config: SFTConfig) -> None:
    report = prepare_sft(config, tokenizer_loader=tokenizer_loader)
    assert report["training_ready"]
    assert report["solution_translation_performed"] is False
    assert all(
        report["audit"][key]
        for key in (
            "matched_original_ids",
            "targets_identical_by_id",
            "balanced_question_languages",
            "train_validation_test_disjoint",
        )
    )
    a = rows(config.output_dir / "english_train.jsonl")
    b = rows(config.output_dir / "multilingual_train.jsonl")
    val = rows(config.output_dir / "validation.jsonl")
    assert len(a) == len(b) == 8 and len(val) == 4
    assert [r["original_problem_id"] for r in a] == [r["original_problem_id"] for r in b]
    assert report["diagnostics"]["multilingual_train"]["question_language_counts"] == {
        "en": 2,
        "zh": 2,
        "de": 2,
        "ja": 2,
    }
    assert report["diagnostics"]["multilingual_train"]["solution_language_counts"] == {"en": 8}
    assert {r["original_problem_id"] for r in a}.isdisjoint(r["original_problem_id"] for r in val)
    for row in a + b + val:
        assert row["solution_language"] == "en" and row["entire_example_multilingual"] is False
        assert row["target_text"] == "Add: 3.5 + 9 = <<3.5+9=12.5>>12.5.\nFINAL_ANSWER: 12.5"
        assert row["messages"][-1] == {"role": "assistant", "content": row["target_text"]}
        assert "worked solution in English" in row["messages"][0]["content"]
        prefix = row["prompt_tokens"]
        assert row["labels"][:prefix] == [-100] * prefix
        assert row["labels"][prefix:] == row["input_ids"][prefix:]
        assert ord("§") in row["labels"][prefix:]
        assert row["token_length"] == len(row["input_ids"]) == len(row["labels"])
        assert row["target_tokens"] + row["prompt_tokens"] == row["token_length"]
        assert not row["formatted_text"].endswith("[assistant]\n")
    for artifact in report["artifacts"].values():
        assert (
            hashlib.sha256((config.output_dir / artifact["path"]).read_bytes()).hexdigest()
            == artifact["sha256"]
        )


def test_deterministic_selection_and_nested_identity(config: SFTConfig) -> None:
    first = prepare_sft(config, tokenizer_loader=tokenizer_loader)
    second_config = replace(config, output_dir=config.output_dir.parent / "repeat")
    second = prepare_sft(second_config, tokenizer_loader=tokenizer_loader)
    assert first["language_assignments"] == second["language_assignments"]
    for name in ("english_train", "multilingual_train", "validation"):
        assert (config.output_dir / f"{name}.jsonl").read_bytes() == (
            second_config.output_dir / f"{name}.jsonl"
        ).read_bytes()
    large = prepare_sft(
        replace(config, train_size=16, output_dir=config.output_dir.parent / "large"),
        tokenizer_loader=tokenizer_loader,
    )
    assert first["selected_train_ids"] == large["selected_train_ids"][:8]
    assert all(
        large["language_assignments"][key] == lang
        for key, lang in first["language_assignments"].items()
    )
    changed = prepare_sft(
        replace(config, seed=43, output_dir=config.output_dir.parent / "changed"),
        tokenizer_loader=tokenizer_loader,
    )
    assert changed["selected_train_ids"] != first["selected_train_ids"]
    with pytest.raises(FileExistsError):
        prepare_sft(config, tokenizer_loader=tokenizer_loader)


def test_overlength_does_not_drop_or_truncate(config: SFTConfig) -> None:
    report = prepare_sft(replace(config, max_seq_length=20), tokenizer_loader=tokenizer_loader)
    assert not report["training_ready"]
    assert report["artifacts"]["issues"]["count"] == 20
    assert len(rows(config.output_dir / "english_train.jsonl")) == 8
    assert len(rows(config.output_dir / "multilingual_train.jsonl")) == 8
    assert report["audit"]["examples_dropped"] == 0
    assert report["audit"]["truncation_performed"] is False
    assert all(r["token_length"] > 20 for r in rows(config.output_dir / "multilingual_train.jsonl"))


def test_mock_questions_block_production_and_label_preview(config: SFTConfig) -> None:
    manifest = json.loads(config.translation_manifest.read_text())
    manifest["mock"] = True
    config.translation_manifest.write_text(json.dumps(manifest))

    def forbidden(_: SFTConfig) -> Any:
        pytest.fail("QA must be checked before tokenizer download")

    with pytest.raises(ValueError, match="mock/unreviewed"):
        prepare_sft(config, tokenizer_loader=forbidden)
    report = prepare_sft(
        replace(config, allow_unreviewed_questions=True), tokenizer_loader=tokenizer_loader
    )
    assert report["status"] == "preview_not_training_ready"
    assert len(report["unreviewed_question_ids"]) == 6


def test_pair_validation_detects_target_edits_duplicates_and_leakage(config: SFTConfig) -> None:
    report = prepare_sft(config, tokenizer_loader=tokenizer_loader)
    a = rows(config.output_dir / "english_train.jsonl")
    b = rows(config.output_dir / "multilingual_train.jsonl")
    val = rows(config.output_dir / "validation.jsonl")
    allowed = set(report["provenance"]["allowed_train_ids"])
    reserved = set(
        report["provenance"]["all_validation_ids"] + report["provenance"]["all_test_ids"]
    )
    with pytest.raises(ValueError, match="Duplicate"):
        validate_sft_pairs(a, b + [b[0]], val, allowed, reserved)
    edited = copy.deepcopy(b)
    edited[0]["target_text"] = edited[0]["target_text"].replace("3.5+9", "3.5-9")
    with pytest.raises(ValueError, match="equations"):
        validate_sft_pairs(a, edited, val, allowed, reserved)
    with pytest.raises(ValueError, match="leakage"):
        validate_sft_pairs(a, b, val, allowed, reserved | {a[0]["original_problem_id"]})
    edited = copy.deepcopy(b)
    edited[0]["solution_language"] = "zh"
    with pytest.raises(ValueError, match="English"):
        validate_sft_pairs(a, edited, val, allowed, reserved)


def change_translations(config: SFTConfig, variant_rows: list[dict]) -> None:
    manifest = json.loads(config.translation_manifest.read_text())
    path = config.translation_manifest.parent / "translations.jsonl"
    content = "".join(json.dumps(row) + "\n" for row in variant_rows)
    path.write_text(content)
    manifest["artifacts"]["variants"].update(
        count=len(variant_rows), sha256=hashlib.sha256(content.encode()).hexdigest()
    )
    config.translation_manifest.write_text(json.dumps(manifest))


def test_translation_identity_and_split_validation(config: SFTConfig) -> None:
    path = config.translation_manifest.parent / "translations.jsonl"
    original = rows(path)
    bad = copy.deepcopy(original)
    bad[0]["split"] = "test"
    change_translations(config, bad)
    with pytest.raises(ValueError, match="split"):
        prepare_sft(config, tokenizer_loader=tokenizer_loader)
    change_translations(config, [row for row in original if row["language"] == "en"])
    with pytest.raises(ValueError, match="Missing assigned"):
        prepare_sft(config, tokenizer_loader=tokenizer_loader)


def test_chat_template_training_contract() -> None:
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "worked solution\nFINAL_ANSWER: 5"},
    ]
    result = tokenize_sft(tokenizer, messages)
    assert tokenizer.calls[0][1] == {"tokenize": False, "add_generation_prompt": False}
    assert tokenizer.calls[1][1] == {"tokenize": True, "add_generation_prompt": False}
    assert tokenizer.calls[2][1] == {"tokenize": False, "add_generation_prompt": True}
    assert tokenizer.calls[2][0] == messages[:-1]
    assert result["target_tokens"] > 0

    class Misaligned(FakeTokenizer):
        def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
            encoded = super().__call__(text, **kwargs)
            if text.endswith("[assistant]\n"):
                encoded["input_ids"][-1] = -123
            return encoded

    with pytest.raises(ValueError, match="token-aligned"):
        tokenize_sft(Misaligned(), messages)


def test_cli_and_config(config: SFTConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    path = config.output_dir.parent / "sft.yaml"
    monkeypatch.setattr(
        cli,
        "prepare_sft",
        lambda conf, **kw: prepare_sft(conf, tokenizer_loader=tokenizer_loader, **kw),
    )
    assert cli.main(["prepare-sft", "--config", str(path)]) == 0
    raw = yaml.safe_load(path.read_text())
    for key, value in (
        ("train_size", 7),
        ("train_size", True),
        ("solution_policy", "translated"),
        ("tokenizer_checkpoint", "other"),
        ("max_seq_length", 40000),
        ("allow_unreviewed_questions", "yes"),
        ("schema_version", 2),
    ):
        path.write_text(yaml.safe_dump({**raw, key: value}))
        with pytest.raises(ValueError):
            load_sft_config(path)
