"""Synthetic paired findings and GPU-free PNG/SVG rendering."""

import csv
import json
import socket

import pytest

from multilingual_llm_reasoning.cli import main
from multilingual_llm_reasoning.comparison_metrics import CONDITIONS, paired_summary
from multilingual_llm_reasoning.error_analysis import CATEGORIES, analyze_results
from multilingual_llm_reasoning.plotting import plot_metrics
from multilingual_llm_reasoning.translation import LANGUAGES


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl-cache"))

    def blocked(*args, **kwargs):
        pytest.fail("Analysis must not access a network")

    monkeypatch.setattr(socket.socket, "connect", blocked)


@pytest.fixture
def inputs(tmp_path):
    expected, rows = [], []
    for i in range(8):
        for j, language in enumerate(LANGUAGES):
            item = {
                "original_problem_id": f"synthetic-{i}",
                "language": language,
                "gold_answer": "2",
                "source_text": f"Synthetic question {i}: 1 + 1?",
                "translated_text": f"{language}: 1 + 1?",
                "qa_status": "source_original" if language == "en" else "human_approved",
                "qa_flags": ["numeric_tokens_changed"] if (i, j) == (0, 1) else [],
            }
            expected.append(item)
            for k, condition in enumerate(CONDITIONS):
                good = (i + j + k) % 4 != 0
                rows.append(
                    {
                        **item,
                        "condition_id": condition,
                        "backend": "mock",
                        "inference_status": "completed",
                        "prediction": "FINAL_ANSWER: 2" if good else "FINAL_ANSWER: 3",
                    }
                )
    # Objective parse evidence, not an automatically assigned reasoning-error category.
    rows[0]["prediction"] = "two"
    for name, values in (("test", expected), ("predictions", rows)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in values))
    report = paired_summary(rows, expected, bootstrap_samples=100)
    report.update(mock=True, research_ready=False)
    (tmp_path / "metrics.json").write_text(json.dumps(report))
    return tmp_path


def test_matched_evidence_and_blank_human_categories(inputs):
    output = inputs / "analysis"
    result = analyze_results(
        inputs / "predictions.jsonl", inputs / "test.jsonl", output, bootstrap_samples=100
    )
    assert result["n_originals"] == 8 and result["n_predictions"] == 96
    assert not result["subjective_categories_assigned"]
    assert result["event_counts"]["cross_language_disagreement"] > 0
    assert result["event_counts"]["correct_to_incorrect"] > 0
    assert result["event_counts"]["incorrect_to_correct"] > 0
    assert result["event_counts"]["translation_flagged"] == 1
    with (output / "human_review.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert all(r["reviewer_category"] == r["reviewer_notes"] == "" for r in rows)
    assert all(tuple(json.loads(r["category_options"])) == CATEGORIES for r in rows)
    evidence = [json.loads(line) for line in (output / "evidence.jsonl").read_text().splitlines()]
    assert {r["review_id"] for r in evidence} == {r["review_id"] for r in rows}
    assert all(r["original_question"] and r["translated_question"] for r in evidence)
    assert any(r["parse_status"] == "missing_marker" for r in evidence)
    # Approved-but-flagged translations must still appear; flags are not truth labels.
    assert any(r["qa_status"] == "human_approved" and r["qa_flags"] for r in evidence)
    again = inputs / "again"
    analyze_results(
        inputs / "predictions.jsonl", inputs / "test.jsonl", again, bootstrap_samples=100
    )
    for file in ("report.json", "events.jsonl", "evidence.jsonl", "human_review.csv"):
        assert (output / file).read_bytes() == (again / file).read_bytes()
    with pytest.raises(ValueError, match="preserve human reviews"):
        analyze_results(inputs / "predictions.jsonl", inputs / "test.jsonl", output)


def test_analysis_rejects_missing_rows(inputs):
    path = inputs / "predictions.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="Missing"):
        analyze_results(path, inputs / "test.jsonl", inputs / "bad", bootstrap_samples=100)
    assert not (inputs / "bad").exists()


def test_render_all_charts_and_loss(inputs):
    pytest.importorskip("matplotlib")
    logs = [
        {"step": 1, "loss": 1.8},
        {"step": 5, "loss": 1.2, "eval_loss": 1.4},
        {"step": 2, "loss": 1.7},
        {"step": 10, "loss": 0.9, "eval_loss": 1.1},
    ]
    log = inputs / "training.jsonl"
    log.write_text("".join(json.dumps(r) + "\n" for r in logs))
    result = plot_metrics(
        inputs / "metrics.json", inputs / "plots", training_logs={"english_lora": log}
    )
    assert result["loss_plot_created"] and len(result["artifacts"]) == 12
    assert not result["warnings"]
    for artifact in result["artifacts"]:
        path = inputs / "plots" / artifact["path"]
        assert path.stat().st_size > 1000
        if path.suffix == ".png":
            assert path.read_bytes().startswith(b"\x89PNG")
        else:
            text = path.read_text()
            assert "<svg" in text and "<text" in text and "MOCK" in text
    assert (inputs / "plots/figures.md").exists()


def test_plot_optional_logs_and_missing_intervals(inputs):
    pytest.importorskip("matplotlib")
    result = plot_metrics(inputs / "metrics.json", inputs / "no-logs")
    assert len(result["artifacts"]) == 10 and not result["loss_plot_created"]
    svg = (inputs / "no-logs/accuracy_by_language.svg").read_text()
    assert "No uncertainty intervals were saved" in svg
    assert (
        "saved paired bootstrap 95%" in (inputs / "no-logs/improvement_by_language.svg").read_text()
    )


def test_invalid_metric_refused_before_output(inputs):
    pytest.importorskip("matplotlib")
    path = inputs / "metrics.json"
    report = json.loads(path.read_text())
    report["summary"][0]["accuracy"] = float("nan")
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="finite"):
        plot_metrics(path, inputs / "bad")
    assert not (inputs / "bad").exists()


def test_cli_analysis(inputs):
    assert (
        main(
            [
                "analyze",
                "--input",
                str(inputs / "predictions.jsonl"),
                "--test-set",
                str(inputs / "test.jsonl"),
                "--output-dir",
                str(inputs / "cli"),
                "--bootstrap-samples",
                "100",
            ]
        )
        == 0
    )
