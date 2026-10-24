"""All source rows are synthetic; no test downloads or reads a Hub dataset."""

import hashlib
import json
import socket
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from multilingual_llm_reasoning import preparation
from multilingual_llm_reasoning.cli import main
from multilingual_llm_reasoning.numeric import normalize_numeric, parse_gsm8k_answer
from multilingual_llm_reasoning.preparation import (
    PreparationConfig,
    _write_jsonl,
    assert_split_integrity,
    load_preparation_config,
    original_problem_id,
    prepare_data,
)

SHA = "a" * 40


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        pytest.fail("Dataset tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1,234.5000", "1234.5"),
        ("+0012.00", "12"),
        ("−1,234.50", "-1234.5"),
        (" -0.000 ", "0"),
        (".50", "0.5"),
        ("-.5", "-0.5"),
        ("1e3", "1000"),
        ("+5E-2", "0.05"),
        ("000", "0"),
        ("1,000", "1000"),
        ("123456789012345678901234567890.12300", "123456789012345678901234567890.123"),
    ],
)
def test_numeric_equivalence(raw: str, expected: str) -> None:
    assert normalize_numeric(raw) == expected
    assert normalize_numeric(expected) == expected


def test_preparation_jsonl_escapes_unicode_line_separators(tmp_path: Path) -> None:
    artifact = _write_jsonl(tmp_path, "rows.jsonl", [{"question": "before\u2028after"}])
    content = (tmp_path / artifact["path"]).read_bytes()
    assert b"\\u2028" in content
    assert len(content.splitlines()) == 1
    assert json.loads(content.splitlines()[0]) == {"question": "before\u2028after"}


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "1,23",
        "12,34,567",
        "1.234,56",
        "1 000",
        "1,000,",
        "1,2,3",
        "1 or 2",
        "1/2",
        "$5",
        "5 dollars",
        "50%",
        "NaN",
        "Infinity",
        "--5",
        "1.2.3",
        "5\n6",
        "1e1001",
        "1e" + "9" * 100,
        "1e-1001",
        "9" * 1001,
        "5.",
    ],
)
def test_ambiguous_or_unsupported_numeric_rejected(raw: str) -> None:
    with pytest.raises(ValueError):
        normalize_numeric(raw)


def test_answer_marker() -> None:
    assert parse_gsm8k_answer("Calculation <<2+3=5>>\n#### +05.00\n") == (
        "Calculation <<2+3=5>>",
        "5",
    )
    for answer in ("5", "#### 5", "reason #### 5 #### 6", "reason #### 5 or 6", None):
        with pytest.raises(ValueError):
            parse_gsm8k_answer(answer)


@pytest.fixture
def source(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[dict[str, str]]]:
    rows = {
        split: [
            {
                "question": f"Synthetic {split} question {index}: what is {index} + 1?",
                "answer": f"Add one.\n#### {index + 1:,}.00",
            }
            for index in range(size)
        ]
        for split, size in (("train", 2200), ("test", 120))
    }
    monkeypatch.setattr(preparation, "load_source", lambda revision: (rows, SHA))
    return rows


@pytest.fixture
def config(tmp_path: Path) -> PreparationConfig:
    return PreparationConfig(
        seed=42,
        validation_size=100,
        train_size=200,
        test_size=100,
        revision=SHA,
        output_dir=tmp_path / "prepared",
    )


def read_rows(root: Path, relative: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (root / relative).read_text().splitlines()]


def ids(rows: list[dict[str, Any]]) -> set[str]:
    return {row["original_problem_id"] for row in rows}


def test_manifest_and_official_test_preserved(source: dict, config: PreparationConfig) -> None:
    manifest = json.loads(prepare_data(config).read_text())
    assert manifest["status"] == "ok"
    assert manifest["dataset"]["resolved_revision"] == SHA
    assert manifest["dataset"]["source_counts"] == {"train": 2200, "test": 120}
    splits = {
        name: read_rows(config.output_dir, artifact["path"])
        for name, artifact in manifest["splits"].items()
    }
    assert_split_integrity(splits)
    assert [row["original_question"] for row in splits["test"]] == [
        row["question"] for row in source["test"]
    ]
    assert len(splits["validation"]) == 100
    assert len(splits["train"]) == 2100
    assert ids(splits["train"]).isdisjoint(ids(splits["validation"]))
    assert ids(splits["test"]).isdisjoint(ids(splits["validation"]) | ids(splits["train"]))
    assert ids(splits["train"]) | ids(splits["validation"]) == {
        original_problem_id(row["question"]) for row in source["train"]
    }
    for split, size in (("train", 200), ("test", 100)):
        subset = read_rows(config.output_dir, manifest["subsets"][split]["path"])
        assert len(subset) == len(ids(subset)) == size
        assert ids(subset) <= ids(splits[split])
    for artifact in [
        *manifest["splits"].values(),
        *manifest["subsets"].values(),
        manifest["errors"],
    ]:
        content = (config.output_dir / artifact["path"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == artifact["sha256"]
        assert len(content.splitlines()) == artifact["count"]
    for split, rows in splits.items():
        for row in rows:
            original = source[row["source_split"]][row["source_index"]]
            assert row["original_question"] == original["question"]
            assert row["original_answer"] == original["answer"]
            assert row["rationale"] == "Add one."
            assert row["gold_answer"] == str(row["source_index"] + 1)
            assert row["split"] == split


def test_reproducible_nested_subsets(source: dict, config: PreparationConfig) -> None:
    prepare_data(config)
    large = replace(config, train_size=2000, output_dir=config.output_dir.parent / "large")
    prepare_data(large)
    for split in ("train", "validation", "test"):
        path = f"splits/{split}.jsonl"
        assert (config.output_dir / path).read_bytes() == (large.output_dir / path).read_bytes()
    assert (
        read_rows(config.output_dir, "subsets/train.jsonl")
        == read_rows(large.output_dir, "subsets/train.jsonl")[:200]
    )
    assert (config.output_dir / "subsets/test.jsonl").read_bytes() == (
        large.output_dir / "subsets/test.jsonl"
    ).read_bytes()
    repeat = replace(config, output_dir=config.output_dir.parent / "repeat")
    prepare_data(repeat)
    first = json.loads((config.output_dir / "manifest.json").read_text())
    second = json.loads((repeat.output_dir / "manifest.json").read_text())
    first["config"].pop("output_dir")
    second["config"].pop("output_dir")
    assert first == second
    changed = replace(config, seed=43, output_dir=config.output_dir.parent / "changed")
    prepare_data(changed)
    assert ids(read_rows(config.output_dir, "splits/validation.jsonl")) != ids(
        read_rows(changed.output_dir, "splits/validation.jsonl")
    )
    with pytest.raises(FileExistsError):
        prepare_data(config)


def test_all_mode(source: dict, config: PreparationConfig) -> None:
    config = replace(config, train_size=None, test_size=None)
    report = json.loads(prepare_data(config).read_text())
    assert report["subsets"]["train"]["count"] == 2100
    assert report["subsets"]["test"]["count"] == 120


def test_invalid_rows_accounted_and_duplicates_quarantined(
    source: dict, config: PreparationConfig
) -> None:
    source["train"][0]["answer"] = "No marker"
    source["test"][0]["answer"] = "Reason #### 1 or 2"
    source["test"][1]["question"] = source["train"][1]["question"]
    source["train"][2]["question"] = ""
    report = json.loads(prepare_data(config).read_text())
    assert report["status"] == "error"
    assert report["subsets"] == {}
    errors = read_rows(config.output_dir, "errors.jsonl")
    assert len(errors) == 5
    assert all(row["error"] for row in errors)
    valid = [
        row
        for split in report["splits"].values()
        for row in read_rows(config.output_dir, split["path"])
    ]
    expected = {(split, index) for split, rows in source.items() for index in range(len(rows))}
    actual = [(row["source_split"], row["source_index"]) for row in valid + errors]
    assert set(actual) == expected
    assert len(actual) == len(expected)
    assert all("gold_answer" in row for row in valid)


@pytest.mark.parametrize(
    "changes", [{"test_size": 121}, {"train_size": 2200}, {"validation_size": 2200}]
)
def test_insufficient_data_is_reported(
    source: dict, config: PreparationConfig, changes: dict
) -> None:
    report = json.loads(prepare_data(replace(config, **changes)).read_text())
    assert report["status"] == "error"
    assert report["failures"]
    assert report["subsets"] == {}


def test_split_assertion_detects_leakage() -> None:
    row = {"original_problem_id": "same", "source_split": "train", "split": "train"}
    with pytest.raises(ValueError, match="repeated"):
        assert_split_integrity({"train": [row], "validation": [{**row, "split": "validation"}]})
    with pytest.raises(ValueError, match="repeated"):
        assert_split_integrity({"train": [row, row]})
    with pytest.raises(ValueError, match="provenance"):
        assert_split_integrity({"test": [row]})


def test_config_and_cli(
    source: dict, config: PreparationConfig, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = {
        "schema_version": 1,
        "seed": 42,
        "validation_size": 100,
        "train_size": 200,
        "test_size": 100,
        "revision": SHA,
        "output_dir": "cli",
    }
    path = tmp_path / "prepare.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert load_preparation_config(path).output_dir == tmp_path / "cli"
    assert main(["prepare-data", "--config", str(path)]) == 0
    assert "manifest.json" in capsys.readouterr().out
    raw["output_dir"] = "bad-cli"
    path.write_text(yaml.safe_dump(raw))
    source["test"][0]["answer"] = "Bad"
    with pytest.raises(SystemExit) as exc:
        main(["prepare-data", "--config", str(path)])
    assert exc.value.code == 2
    assert (tmp_path / "bad-cli/errors.jsonl").exists()
    for key, value in (
        ("seed", True),
        ("train_size", 0),
        ("test_size", -1),
        ("validation_size", 0),
        ("schema_version", True),
        ("unknown", 3),
    ):
        path.write_text(yaml.safe_dump({**raw, key: value}))
        with pytest.raises(ValueError):
            load_preparation_config(path)


def test_hugging_face_loader_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    fake_source = {"train": [], "test": []}

    def load_dataset(*args: object, **kwargs: object) -> dict:
        calls.append((args, kwargs))
        return fake_source

    class FakeApi:
        def dataset_info(self, name: str, *, revision: str) -> SimpleNamespace:
            assert name == "openai/gsm8k"
            assert revision == "main"
            return SimpleNamespace(sha=SHA)

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeApi))
    assert preparation.load_source("main") == (fake_source, SHA)
    assert preparation.load_source(SHA) == (fake_source, SHA)
    assert calls == [(("openai/gsm8k", "main"), {"revision": SHA})] * 2


def test_membership_independent_of_source_order(source: dict, config: PreparationConfig) -> None:
    prepare_data(config)
    reordered = replace(config, output_dir=config.output_dir.parent / "reordered")
    for rows in source.values():
        rows.reverse()
    prepare_data(reordered)
    for relative in (
        "splits/train.jsonl",
        "splits/validation.jsonl",
        "splits/test.jsonl",
        "subsets/train.jsonl",
        "subsets/test.jsonl",
    ):
        assert ids(read_rows(config.output_dir, relative)) == ids(
            read_rows(reordered.output_dir, relative)
        )
