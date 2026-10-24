"""Unicode separators inside JSON strings must not become physical JSONL records."""

import json

import pytest

from multilingual_llm_reasoning.baseline_scoring import score_file
from multilingual_llm_reasoning.datasets import load_problems
from multilingual_llm_reasoning.error_analysis import read_jsonl


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\u0085"])
def test_unicode_question_survives_ingestion_and_analysis(tmp_path, separator):
    text = f"First sentence.{separator}Second sentence."
    record = {"id": "original-1", "language": "en", "prompt": text, "answer": "2"}
    path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    assert load_problems(path, ("en",))[0].prompt == text
    assert read_jsonl(path) == [record]


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\u0085"])
def test_unicode_evidence_does_not_inflate_scoring_denominator(tmp_path, separator):
    record = {
        "original_problem_id": "original-1",
        "language": "ja",
        "model_checkpoint": "mock",
        "model_revision": "mock-v1",
        "question": f"Question{separator}continued",
        "prediction": "FINAL_ANSWER: 2",
        "gold_answer": "2",
    }
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    output = tmp_path / "scored"
    score_file(path, output)
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["overall"]["total"] == 1
    assert metrics["overall"]["correct"] == 1
    assert metrics["overall"]["invalid_records"] == 0
