"""Single-condition baseline metrics and plots; synthetic records only."""

import hashlib
import json
from pathlib import Path

import pytest

from multilingual_llm_reasoning.baseline_analysis import analyze_baseline
from multilingual_llm_reasoning.plotting import plot_baseline_metrics
from multilingual_llm_reasoning.scoring import PARSER_VERSION


@pytest.fixture
def baseline_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    variants = []
    predictions = []
    settings = {
        "backend": "transformers",
        "model_checkpoint": "Qwen/Qwen2.5-3B-Instruct",
        "model_revision": "pinned-revision",
        "prompt_template_version": "prompt-v1",
        "chat_template_sha256": "template-hash",
        "decoding": {"do_sample": False, "temperature": 0, "max_new_tokens": 256},
        "parser_version": PARSER_VERSION,
    }
    outcomes = {
        ("id-a", "en"): "1",
        ("id-a", "zh"): "1",
        ("id-a", "de"): "1",
        ("id-a", "ja"): "1",
        ("id-b", "en"): "2",
        ("id-b", "zh"): "3",
        ("id-b", "de"): "malformed",
        ("id-b", "ja"): "2",
    }
    for identity, gold in (("id-a", "1"), ("id-b", "2")):
        for language in ("en", "zh", "de", "ja"):
            variants.append(
                {
                    "original_problem_id": identity,
                    "language": language,
                    "gold_answer": gold,
                    "source_text": f"question {identity}",
                    "translated_text": f"{language} question {identity}",
                    "split": "test",
                    "source_split": "test",
                    "qa_status": "source_original" if language == "en" else "human_approved",
                    "qa_flags": [],
                }
            )
            answer = outcomes[identity, language]
            prediction = "No answer marker" if answer == "malformed" else f"FINAL_ANSWER: {answer}"
            predictions.append(
                {
                    **settings,
                    "original_problem_id": identity,
                    "language": language,
                    "gold_answer": gold,
                    "prediction": prediction,
                    "inference_status": "completed",
                }
            )
    test_path = tmp_path / "variants.jsonl"
    test_path.write_text("".join(json.dumps(row) + "\n" for row in variants))
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text("".join(json.dumps(row) + "\n" for row in predictions))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "complete": True,
                "research_ready": True,
                "mock": False,
                "artifacts": {
                    "variants": {
                        "path": test_path.name,
                        "sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
                    }
                },
            }
        )
    )
    return predictions_path, test_path, manifest_path


def test_baseline_analysis_keeps_paired_denominators_and_reports_disagreement(
    baseline_fixture: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    predictions, test_set, manifest = baseline_fixture
    report = analyze_baseline(predictions, test_set, manifest, tmp_path / "analysis")
    assert report["n_originals"] == 2
    assert report["n_predictions"] == 8
    assert report["parser_version"] == PARSER_VERSION
    assert report["language_metrics"]["de"]["n_expected"] == 2
    assert report["language_metrics"]["de"]["accuracy"] == 0.5
    assert report["language_metrics"]["de"]["parse_failures"] == 1
    assert report["english_minus_other_language_accuracy_gaps"]["zh"] == 0.5
    assert report["cross_language_consistency"]["all_four_correct"] == 0.5
    assert report["disagreement_count"] == 1
    disagreement = json.loads(
        (tmp_path / "analysis/disagreements.jsonl").read_text().splitlines()[0]
    )
    assert disagreement["original_problem_id"] == "id-b"


def test_baseline_analysis_rejects_missing_predictions(
    baseline_fixture: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    predictions, test_set, manifest = baseline_fixture
    rows = predictions.read_text().splitlines()[:-1]
    predictions.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="incomplete"):
        analyze_baseline(predictions, test_set, manifest, tmp_path / "analysis")


def test_baseline_figures_export_png_svg(
    baseline_fixture: tuple[Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    predictions, test_set, manifest = baseline_fixture
    analysis = tmp_path / "analysis"
    analyze_baseline(predictions, test_set, manifest, analysis)
    result = plot_baseline_metrics(analysis / "summary.json", tmp_path / "figures")
    assert len(result["artifacts"]) == 8
    assert all(
        (tmp_path / "figures" / artifact["path"]).stat().st_size > 0
        for artifact in result["artifacts"]
    )
    assert (tmp_path / "figures/figures.md").is_file()
