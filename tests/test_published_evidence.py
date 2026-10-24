"""Verify the publishable Phase 16L aggregates without loading models or raw prompts."""

import csv
import hashlib
import json
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parents[1] / "docs/evidence/phase16l"


def test_phase16l_published_aggregate_hashes_and_denominators():
    manifest = json.loads((EVIDENCE / "reproducibility_manifest.json").read_text())
    for name in (
        "summary.json",
        "summary.csv",
        "overall_accuracy.csv",
        "parsing_failures.csv",
        "contrasts.csv",
        "mcnemar.csv",
        "english_gaps.csv",
        "consistency.csv",
        "statistical_comparison.md",
        "final_evaluation_report.md",
    ):
        path = EVIDENCE / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["artifact_sha256"][name]
    summary = json.loads((EVIDENCE / "summary.json").read_text())
    assert summary["complete"] and not summary["mock"] and not summary["research_ready"]
    assert summary["n_originals"] == 98 and summary["expected_predictions"] == 1176
    assert summary["baseline_reused"]
    rows = list(csv.DictReader((EVIDENCE / "overall_accuracy.csv").open(newline="")))
    assert {(r["condition_id"], int(r["n_correct"]), int(r["n_parse_failures"])) for r in rows} == {
        ("base", 273, 48),
        ("english_lora", 195, 26),
        ("multilingual_lora", 208, 19),
    }
    assert {int(r["n_expected"]) for r in rows} == {392}
    for row in rows:
        assert float(row["accuracy"]) == int(row["n_correct"]) / 392
        groups = [r for r in summary["summary"] if r["condition_id"] == row["condition_id"]]
        assert len(groups) == 4
        assert sum(r["n_correct"] for r in groups) == int(row["n_correct"])
        assert sum(r["n_parse_failures"] for r in groups) == int(row["n_parse_failures"])
        assert {r["n_expected"] for r in groups} == {98}
