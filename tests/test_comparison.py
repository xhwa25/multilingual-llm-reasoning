"""Paired metrics and adapter selection without model downloads or GPU execution."""

import copy
import json
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_baseline import config as baseline_fixture  # noqa: F401

from multilingual_llm_reasoning.comparison import evaluate_comparison, load_comparison_config
from multilingual_llm_reasoning.comparison_backends import activate_adapter, audit_adapter
from multilingual_llm_reasoning.comparison_metrics import CONDITIONS, paired_summary
from multilingual_llm_reasoning.translation import LANGUAGES

ROOT = Path(__file__).resolve().parents[1]


def fixture_rows():
    expected = [
        {"original_problem_id": f"p{i}", "language": lang, "gold_answer": "2"}
        for i in range(4)
        for lang in LANGUAGES
    ]
    rows = [
        {**r, "condition_id": c, "inference_status": "completed", "prediction": "FINAL_ANSWER: 2"}
        for c in CONDITIONS
        for r in expected
    ]
    return expected, rows


def test_metrics_denominators_consistency_and_pairing():
    expected, rows = fixture_rows()
    base_zh = [r for r in rows if r["condition_id"] == "base" and r["language"] == "zh"]
    base_zh[0]["prediction"] = "cannot parse"
    base_zh[1].update(inference_status="error", prediction=None)
    report = paired_summary(rows, expected, bootstrap_samples=200)
    zh = next(r for r in report["summary"] if r["condition_id"] == "base" and r["language"] == "zh")
    assert zh["n_expected"] == 4 and zh["n_completed"] == 3 and zh["n_correct"] == 2
    assert zh["accuracy"] == 0.5 and zh["parse_failure_rate"] == 0.25
    assert zh["parse_failure_rate_among_completed"] == 1 / 3
    assert zh["n_inference_failures_or_pending"] == 1
    gap = next(
        r
        for r in report["english_gaps"]
        if r["condition_id"] == "base" and r["other_language"] == "zh"
    )
    assert gap["estimate"] == 0.5
    consistency = next(r for r in report["consistency"] if r["condition_id"] == "base")
    assert consistency["estimate"] == 0.5 and consistency["n_originals"] == 4
    contrast = next(
        r
        for r in report["contrasts"]
        if r["condition_id"] == "english_lora" and r["reference"] == "base" and r["metric"] == "zh"
    )
    assert contrast["estimate"] == 0.5
    assert contrast["ci95_low"] <= 0.5 <= contrast["ci95_high"]
    random.Random(4).shuffle(rows)
    assert paired_summary(rows, list(reversed(expected)), bootstrap_samples=200) == report


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("duplicate", "Duplicate"),
        ("missing", "Missing"),
        ("gold", "Gold mismatch"),
        ("extra", "Unexpected"),
        ("prompt", "Unpaired"),
    ],
)
def test_strict_joins(mutation, match):
    expected, rows = fixture_rows()
    if mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    if mutation == "missing":
        rows.pop()
    if mutation == "gold":
        rows[0]["gold_answer"] = "3"
    if mutation == "extra":
        rows[0]["condition_id"] = "other"
    if mutation == "prompt":
        rows[0]["prompt_messages"] = [{"role": "user", "content": "different"}]
    with pytest.raises(ValueError, match=match):
        paired_summary(rows, expected, bootstrap_samples=100)


def test_bootstrap_is_paired_not_independent():
    expected, rows = fixture_rows()
    for row in rows:
        if row["original_problem_id"] in ("p0", "p2"):
            row["prediction"] = "FINAL_ANSWER: 7"
        row["correct"] = True  # Re-scoring must disregard this stale flag.
    report = paired_summary(rows, expected, bootstrap_samples=200)
    assert all(r["estimate"] == r["ci95_low"] == r["ci95_high"] == 0 for r in report["contrasts"])
    assert all(r["accuracy"] == 0.5 for r in report["summary"])


@dataclass
class Status:
    enabled: bool = True
    active_adapters: object = None
    merged_adapters: object = None
    num_adapter_layers: int = 2


def test_adapter_activation_checked(monkeypatch, tmp_path):
    status = Status(active_adapters=["english_lora"], merged_adapters=[])
    seen = {}
    wrapper = SimpleNamespace(
        set_adapter=lambda name: seen.update(active=name),
        get_model_status=lambda: status,
        peft_config={"english_lora": {}},
        requires_grad_=lambda value: seen.update(grad=value),
        eval=lambda: seen.update(eval=True),
    )

    def load(model, path, **kwargs):
        seen.update(kwargs)
        return wrapper

    monkeypatch.setitem(
        sys.modules, "peft", SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=load))
    )
    _, audit = activate_adapter(object(), tmp_path, "english_lora")
    assert seen == {
        "adapter_name": "english_lora",
        "is_trainable": False,
        "active": "english_lora",
        "grad": False,
        "eval": True,
    }
    assert audit["active_adapters"] == ["english_lora"]
    status.active_adapters = ["multilingual_lora"]
    with pytest.raises(ValueError, match="activation"):
        activate_adapter(object(), tmp_path, "english_lora")


def test_adapter_provenance_leakage(tmp_path):
    cfg = load_comparison_config(ROOT / "configs/comparison.yaml").baseline
    meta = {
        "condition": "english_train",
        "model": cfg.model,
        "revision": cfg.revision,
        "train_ids": ["train"],
        "validation_ids": ["val"],
    }
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "base_model_name_or_path": cfg.model})
    )
    (tmp_path / "adapter_model.safetensors").write_bytes(b"fixture-only")
    (tmp_path / "evaluation_provenance.json").write_text(json.dumps(meta))
    assert len(audit_adapter(tmp_path, "english_lora", cfg, {"test"})["files"]) == 3
    with pytest.raises(ValueError, match="leakage"):
        audit_adapter(tmp_path, "english_lora", cfg, {"train"})
    with pytest.raises(ValueError, match="condition"):
        audit_adapter(tmp_path, "multilingual_lora", cfg, {"test"})


def test_comparison_mock_resume(tmp_path, monkeypatch, baseline_fixture):  # noqa: F811
    baseline = baseline_fixture
    cfg = load_comparison_config(ROOT / "configs/comparison_mock.yaml")
    cfg = replace(
        cfg, baseline=replace(baseline, output_dir=tmp_path / "comparison"), bootstrap_samples=100
    )
    report = evaluate_comparison(cfg)
    assert len(report["summary"]) == 12 and report["mock"] and not report["research_ready"]
    from multilingual_llm_reasoning.baseline_backends import MockBaselineBackend

    monkeypatch.setattr(
        MockBaselineBackend, "generate", lambda *a: pytest.fail("resume generated again")
    )
    assert evaluate_comparison(cfg) == report
    rows = [
        json.loads(line)
        for line in (cfg.baseline.output_dir / "predictions.jsonl").read_text().splitlines()
    ]
    assert {row["condition_id"] for row in rows} == set(CONDITIONS)
    assert len(rows) == 12 * cfg.baseline.num_questions


def test_cluster_bootstrap_keeps_languages_together():
    expected, rows = fixture_rows()
    for row in rows:
        correct = row["language"] in ("en", "zh")
        if row["condition_id"] != "base":
            correct = not correct
        row["prediction"] = "FINAL_ANSWER: 2" if correct else "FINAL_ANSWER: 3"
    report = paired_summary(rows, expected, bootstrap_samples=100)
    macro = [r for r in report["contrasts"] if r["metric"] == "macro_accuracy"]
    assert all(r["estimate"] == r["ci95_low"] == r["ci95_high"] == 0 for r in macro)


def test_offline_summary_cli(tmp_path):
    from multilingual_llm_reasoning.cli import main

    expected, rows = fixture_rows()
    for name, values in (("predictions", rows), ("expected", expected)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in values))
    output = tmp_path / "summary"
    assert (
        main(
            [
                "summarize-comparison",
                "--input",
                str(tmp_path / "predictions.jsonl"),
                "--test-set",
                str(tmp_path / "expected.jsonl"),
                "--output-dir",
                str(output),
                "--bootstrap-samples",
                "100",
            ]
        )
        == 0
    )
    import csv

    with (output / "summary.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 12


def test_initialization_errors_retained_with_condition(tmp_path, baseline_fixture, monkeypatch):  # noqa: F811
    import multilingual_llm_reasoning.comparison as comparison

    cfg = load_comparison_config(ROOT / "configs/comparison_mock.yaml")
    cfg = replace(
        cfg,
        baseline=replace(baseline_fixture, output_dir=tmp_path / "failed"),
        bootstrap_samples=100,
    )

    def fail(*args):
        raise RuntimeError("fixture initialization failure")

    monkeypatch.setattr(comparison, "MockBaselineBackend", fail)
    report = evaluate_comparison(cfg)
    assert not report["complete"] and not report["research_ready"]
    assert all(
        r["n_completed"] == 0 and r["n_expected"] == cfg.baseline.num_questions
        for r in report["summary"]
    )
    for condition in CONDITIONS:
        attempts = cfg.baseline.output_dir / condition / "inference_attempts.jsonl"
        rows = [json.loads(line) for line in attempts.read_text().splitlines()]
        assert len(rows) == cfg.baseline.num_questions * 4
        assert all(
            row["condition_id"] == condition and row["parse_status"] == "inference_error"
            for row in rows
        )
