"""Phase 16L offline contract, paired tests, and crash-safe persistence checks."""

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from multilingual_llm_reasoning.baseline import load_baseline_inputs
from multilingual_llm_reasoning.comparison import (
    load_comparison_config,
    verify_baseline_reuse,
)
from multilingual_llm_reasoning.comparison_metrics import exact_mcnemar
from scripts.phase16l_evaluation_runner import restore_progress, sqlite_backup, write_final_reports

ROOT = Path(__file__).resolve().parents[1]


def test_saved_real_baseline_exact_reuse_contract(private_final_run_inputs: None):
    config = load_comparison_config(ROOT / "configs/phase16l_comparison.yaml")
    inputs, source_hash, reviewed = load_baseline_inputs(config.baseline)
    rows, manifest = verify_baseline_reuse(
        config.baseline_reuse_manifest, config.baseline, inputs, source_hash
    )
    assert not reviewed and not manifest["mock_model"]
    assert len(rows) == len(inputs) == 392
    assert len({r["original_problem_id"] for r in rows}) == 98
    changed = replace(config.baseline, max_new_tokens=128)
    with pytest.raises(ValueError, match="max_new_tokens"):
        verify_baseline_reuse(config.baseline_reuse_manifest, changed, inputs, source_hash)
    with pytest.raises(ValueError, match="different translations"):
        verify_baseline_reuse(config.baseline_reuse_manifest, config.baseline, inputs, "bad")


def test_frozen_sft_ids_and_hashes_do_not_touch_held_out_grid(private_final_run_inputs: None):
    import hashlib

    config = load_comparison_config(ROOT / "configs/phase16l_comparison.yaml")
    inputs, _, _ = load_baseline_inputs(config.baseline)
    test_ids = {row["original_problem_id"] for row in inputs}
    frozen = ROOT / "outputs/phase16h-approved-sft-freeze-v1"
    manifest = json.loads((frozen / "dataset_freeze_manifest.json").read_text())
    expected_ids = set(manifest["canonical_ids"])
    assert len(expected_ids) == 197 and not expected_ids & test_ids
    for name in ("english_only_sft_final.jsonl", "multilingual_sft_final.jsonl"):
        path = frozen / name
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest() == manifest["dataset_files_sha256"][name]
        )
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 197
        assert {row["original_problem_id"] for row in rows} == expected_ids


def test_exact_mcnemar_binary_pairing():
    assert exact_mcnemar([1, 1, 1, 0], [0, 0, 0, 0]) == {
        "left_only_correct": 3,
        "right_only_correct": 0,
        "discordant_pairs": 3,
        "p_value_two_sided": 0.25,
    }
    assert exact_mcnemar([1, 0], [1, 0])["p_value_two_sided"] == 1.0
    with pytest.raises(ValueError, match="equal-length"):
        exact_mcnemar([1], [1, 0])


def test_sqlite_online_backup_and_restore(tmp_path):
    source = tmp_path / "live.sqlite3"
    with sqlite3.connect(source) as db:
        db.execute("CREATE TABLE attempts (payload TEXT NOT NULL)")
        db.execute(
            "INSERT INTO attempts(payload) VALUES (?)",
            (json.dumps({"inference_status": "completed"}),),
        )
        db.commit()
        backup = tmp_path / "drive/progress/english_lora/baseline.sqlite3"
        backup.parent.mkdir(parents=True)
        assert sqlite_backup(source, backup) == 1
        assert backup.is_file()
    local = tmp_path / "new_local"
    restore_progress(local, tmp_path / "drive")
    restored = local / "english_lora/baseline.sqlite3"
    with sqlite3.connect(restored) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 1
    with pytest.raises(ValueError, match="nonempty"):
        restore_progress(local, tmp_path / "drive")


def test_notebook_evaluation_switches_default_off():
    notebook = json.loads((ROOT / "notebooks/phase16l_colab_final_evaluation.ipynb").read_text())
    code = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    for switch in (
        "VERIFY_ADAPTER_LOAD",
        "RUN_ENGLISH_EVAL",
        "RUN_MULTILINGUAL_EVAL",
        "RUN_FINALIZE",
        "RUN_OFFLINE_REPORTS",
    ):
        assert f"{switch} = False" in code
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "notebook", "exec")


def test_final_reports_use_saved_denominators(tmp_path):
    summary = {
        "complete": True,
        "expected_predictions": 1176,
        "summary": [
            {
                "condition_id": condition,
                "language": language,
                "n_expected": 98,
                "n_correct": 49,
                "n_parse_failures": 2,
                "parse_failure_rate": 2 / 98,
            }
            for condition in ("base", "english_lora", "multilingual_lora")
            for language in ("en", "zh", "de", "ja")
        ],
        "denominator_policy": "Full expected denominator",
        "bootstrap": {"method": "paired bootstrap", "resamples": 5000},
        "mcnemar_scope": "Per-language exact test",
        "uncertainty": "Exploratory",
        "contrasts": [
            {
                "condition_id": "multilingual_lora",
                "reference": "english_lora",
                "metric": "en",
                "estimate": 0.0,
                "ci95_low": -0.1,
                "ci95_high": 0.1,
            }
        ],
        "mcnemar": [
            {
                "condition_id": "multilingual_lora",
                "reference": "english_lora",
                "language": "en",
                "discordant_pairs": 4,
                "p_value_two_sided": 1.0,
            }
        ],
    }
    write_final_reports(tmp_path, summary)
    import csv

    with (tmp_path / "overall_accuracy.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert {row["n_expected"] for row in rows} == {"392"}
    assert {row["n_correct"] for row in rows} == {"196"}
    assert {row["n_parse_failures"] for row in rows} == {"8"}
    for name in ("parsing_failures.csv", "statistical_comparison.md", "final_evaluation_report.md"):
        assert (tmp_path / name).stat().st_size > 0
    summary["summary"][0]["n_expected"] = 97
    with pytest.raises(ValueError, match="denominators"):
        write_final_reports(tmp_path, summary)
