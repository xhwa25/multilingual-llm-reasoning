"""Guarded Colab comparison launcher with periodic SQLite backups to Google Drive."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

from multilingual_llm_reasoning.baseline import load_baseline_inputs
from multilingual_llm_reasoning.comparison import (
    load_comparison_config,
    verify_baseline_reuse,
)
from multilingual_llm_reasoning.comparison_backends import audit_adapter

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/phase16l_comparison.yaml"
CONDITIONS = ("english_lora", "multilingual_lora")
FROZEN = ROOT / "outputs/phase16h-approved-sft-freeze-v1"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def preflight() -> tuple[object, dict]:
    config = load_comparison_config(CONFIG_PATH)
    inputs, source_hash, reviewed = load_baseline_inputs(config.baseline)
    if len(inputs) != 392 or len({r["original_problem_id"] for r in inputs}) != 98:
        raise ValueError("Phase 16L requires exactly 98 originals and 392 aligned variants")
    if reviewed:
        raise ValueError("This Phase 16L config expects the recorded provisional dataset status")
    baseline_rows, _ = verify_baseline_reuse(
        config.baseline_reuse_manifest, config.baseline, inputs, source_hash
    )
    if len(baseline_rows) != 392:
        raise ValueError("Baseline reuse is incomplete")
    ids = {r["original_problem_id"] for r in inputs}
    freeze = json.loads((FROZEN / "dataset_freeze_manifest.json").read_text())
    if not freeze.get("training_ready") or freeze.get("model_revision") != config.baseline.revision:
        raise ValueError(
            "Frozen SFT manifest is not training-ready or has a different base revision"
        )
    frozen_ids = set(freeze["canonical_ids"])
    if len(frozen_ids) != 197 or frozen_ids & ids:
        raise ValueError("Frozen SFT canonical IDs are duplicated or overlap evaluation")
    for name in ("english_only_sft_final.jsonl", "multilingual_sft_final.jsonl"):
        path = FROZEN / name
        if digest(path) != freeze["dataset_files_sha256"][name]:
            raise ValueError(f"Frozen SFT dataset hash mismatch: {name}")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != 197 or {row["original_problem_id"] for row in rows} != frozen_ids:
            raise ValueError(f"Frozen SFT canonical grid changed: {name}")
    adapters = {
        "english_lora": audit_adapter(config.english_adapter, "english_lora", config.baseline, ids),
        "multilingual_lora": audit_adapter(
            config.multilingual_adapter, "multilingual_lora", config.baseline, ids
        ),
    }
    a, b = [adapters[name]["provenance"] for name in CONDITIONS]
    for field in ("train_ids", "validation_ids", "training_budget"):
        if a[field] != b[field]:
            raise ValueError(f"Adapter conditions differ in {field}")
    if set(a["train_ids"]) != frozen_ids:
        raise ValueError("Adapter training IDs do not match the approved frozen SFT datasets")
    if set(a["train_ids"]) & ids or set(a["validation_ids"]) & ids:
        raise ValueError("Training/evaluation canonical-ID overlap")
    contract = {
        "schema_version": 1,
        "dataset_version": "phase14i-lite-provisional-dataset-v1",
        "research_ready": False,
        "n_originals": 98,
        "n_variants": 392,
        "expected_responses": 1176,
        "translation_manifest_sha256": source_hash,
        "variants_sha256": json.loads(config.baseline.translation_manifest.read_text())[
            "artifacts"
        ]["variants"]["sha256"],
        "baseline_reuse_manifest_sha256": digest(config.baseline_reuse_manifest),
        "baseline_predictions_sha256": json.loads(config.baseline_reuse_manifest.read_text())[
            "artifacts"
        ]["predictions"]["sha256"],
        "config_sha256": {
            "comparison": digest(CONFIG_PATH),
            "baseline": digest(ROOT / "configs/phase16l_baseline.yaml"),
        },
        "adapter_files_sha256": {name: adapters[name]["files"] for name in CONDITIONS},
        "frozen_sft_sha256": {
            name: freeze["dataset_files_sha256"][name]
            for name in ("english_only_sft_final.jsonl", "multilingual_sft_final.jsonl")
        },
        "model_revision": config.baseline.revision,
        "parser_version": baseline_rows[0]["parser_version"],
        "baseline_device": baseline_rows[0]["device"],
        "adapter_device": config.baseline.device,
        "bootstrap_unit": "original_problem_id",
    }
    return config, contract


def initialize_destination(drive_root: Path, contract: dict, *, resume: bool) -> None:
    marker = drive_root / "evaluation_contract.json"
    if not drive_root.exists():
        if resume:
            raise FileNotFoundError("Drive comparison destination does not exist")
        drive_root.mkdir(parents=True)
        marker.write_text(json.dumps(contract, indent=2) + "\n")
    elif not marker.is_file() or json.loads(marker.read_text()) != contract:
        raise ValueError("Drive comparison contract missing or changed")


def sqlite_backup(source: Path, destination: Path) -> int:
    """Use SQLite's online backup API; never copy a live database file directly."""
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.partial")
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as reader:
        with sqlite3.connect(temporary) as writer:
            reader.backup(writer)
            if writer.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("SQLite backup integrity check failed")
            count = sum(
                json.loads(row[0]).get("inference_status") == "completed"
                for row in writer.execute("SELECT payload FROM attempts")
            )
    temporary.replace(destination)
    return int(count)


def persist_progress(local_root: Path, drive_root: Path) -> dict:
    counts = {}
    for condition in CONDITIONS:
        source = local_root / condition / "baseline.sqlite3"
        if not source.is_file():
            counts[condition] = 0
            continue
        target_dir = drive_root / "progress" / condition
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            counts[condition] = sqlite_backup(source, target_dir / "baseline.sqlite3")
        except sqlite3.OperationalError as exc:
            if "no such table" not in str(exc):
                raise
            counts[condition] = 0  # Database just created; schema is still initializing.
    progress = {"completed_attempts": counts, "total_expected_per_adapter": 392}
    temporary = drive_root / f".progress.{uuid4().hex}.partial"
    temporary.write_text(json.dumps(progress, indent=2) + "\n")
    temporary.replace(drive_root / "progress.json")
    return progress


def restore_progress(local_root: Path, drive_root: Path) -> None:
    if local_root.exists() and any(local_root.iterdir()):
        raise ValueError("Local comparison output is nonempty; do not overwrite it")
    for condition in CONDITIONS:
        source = drive_root / "progress" / condition / "baseline.sqlite3"
        if not source.is_file():
            continue
        with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as database:
            if database.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("Saved SQLite progress is corrupt")
        target = local_root / condition / "baseline.sqlite3"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def write_final_reports(local_root: Path, summary: dict) -> None:
    """Create human-readable tables only from a complete, saved comparison."""
    if not summary.get("complete") or summary.get("expected_predictions") != 1176:
        raise ValueError("Cannot report an incomplete Phase 16L comparison")
    groups = summary["summary"]
    if len(groups) != 12 or {r["n_expected"] for r in groups} != {98}:
        raise ValueError("Phase 16L summary has changed denominators")
    condition_rows = []
    for condition in ("base", *CONDITIONS):
        matching = [r for r in groups if r["condition_id"] == condition]
        if {r["language"] for r in matching} != {"en", "zh", "de", "ja"}:
            raise ValueError("Phase 16L summary is missing language groups")
        n = sum(r["n_expected"] for r in matching)
        correct = sum(r["n_correct"] for r in matching)
        parse_failures = sum(r["n_parse_failures"] for r in matching)
        condition_rows.append(
            {
                "condition_id": condition,
                "n_expected": n,
                "n_correct": correct,
                "accuracy": correct / n,
                "n_parse_failures": parse_failures,
                "parse_failure_rate": parse_failures / n,
            }
        )

    def write_csv(name: str, rows: list[dict]) -> None:
        with (local_root / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv("overall_accuracy.csv", condition_rows)
    write_csv(
        "parsing_failures.csv",
        [
            {
                "condition_id": r["condition_id"],
                "language": r["language"],
                "n_expected": r["n_expected"],
                "n_parse_failures": r["n_parse_failures"],
                "parse_failure_rate": r["parse_failure_rate"],
            }
            for r in groups
        ],
    )
    statistics = [
        "# Phase 16L paired comparisons — provisional",
        "",
        summary["denominator_policy"],
        "",
        summary["bootstrap"]["method"]
        + f"; {summary['bootstrap']['resamples']} resamples; unit: original problem ID.",
        summary["mcnemar_scope"],
        summary["uncertainty"],
        "",
        "| Treatment | Reference | Metric | Difference | 95% CI |",
        "|---|---|---|---:|---:|",
    ]
    for row in summary["contrasts"]:
        statistics.append(
            f"| {row['condition_id']} | {row['reference']} | {row['metric']} | "
            f"{row['estimate']:.3f} | [{row['ci95_low']:.3f}, {row['ci95_high']:.3f}] |"
        )
    statistics += [
        "",
        "| Treatment | Reference | Language | Discordant | Exact two-sided p |",
        "|---|---|---|---:|---:|",
    ]
    for row in summary["mcnemar"]:
        statistics.append(
            f"| {row['condition_id']} | {row['reference']} | {row['language']} | "
            f"{row['discordant_pairs']} | {row['p_value_two_sided']:.4f} |"
        )
    (local_root / "statistical_comparison.md").write_text("\n".join(statistics) + "\n")
    report = [
        "# Phase 16L held-out multilingual evaluation — provisional",
        "",
        "The three conditions share 98 canonical GSM8K test questions and four language variants "
        "per question (1,176 responses). The translation dataset has limited QA, pending human "
        "approval, and research_ready: false.",
        "",
        "The base predictions were generated on MPS FP16; the LoRA predictions use CUDA FP16. "
        "Model revision, prompt, decoding, parser, and exact test inputs were audited for reuse. "
        "Cross-device numerical differences remain possible.",
        "",
        "| Condition | Correct / 392 | Accuracy | Parse failures / 392 |",
        "|---|---:|---:|---:|",
    ]
    for row in condition_rows:
        report.append(
            f"| {row['condition_id']} | {row['n_correct']}/{row['n_expected']} | "
            f"{row['accuracy']:.3f} | {row['n_parse_failures']}/{row['n_expected']} |"
        )
    report += [
        "",
        "Per-language denominators, paired uncertainty, and exact McNemar results are in the "
        "saved CSV/JSON files and statistical_comparison.md. Parse failures count incorrect. "
        "The instruction wrapper is English across all languages. These exploratory results "
        "do not establish a fine-tuning benefit without interpreting uncertainty and "
        "translation risks.",
        "",
    ]
    (local_root / "final_evaluation_report.md").write_text("\n".join(report))


def persist_final(local_root: Path, drive_root: Path, contract: dict) -> None:
    summary = json.loads((local_root / "summary.json").read_text())
    if not summary.get("complete") or summary.get("research_ready") is not False:
        raise ValueError("Comparison is incomplete or provisional status was lost")
    if summary.get("expected_predictions") != 1176:
        raise ValueError("Comparison prediction denominator changed")
    write_final_reports(local_root, summary)
    final = drive_root / "final"
    if final.exists():
        raise FileExistsError("Final Drive evaluation output already exists")
    temporary = drive_root / f".final.{uuid4().hex}.partial"
    temporary.mkdir()
    required = (
        "summary.json",
        "summary.csv",
        "contrasts.csv",
        "mcnemar.csv",
        "english_gaps.csv",
        "consistency.csv",
        "predictions.jsonl",
        "selected_test.jsonl",
        "comparison_contract.json",
        "overall_accuracy.csv",
        "parsing_failures.csv",
        "statistical_comparison.md",
        "final_evaluation_report.md",
    )
    for name in required:
        source = local_root / name
        if not source.is_file() or source.stat().st_size == 0:
            raise ValueError(f"Missing final comparison artifact: {source}")
        shutil.copy2(source, temporary / name)
    for condition in ("base", *CONDITIONS):
        source = local_root / condition / "predictions.jsonl"
        target = temporary / condition
        target.mkdir()
        shutil.copy2(source, target / "predictions.jsonl")
    hashes = {
        str(path.relative_to(temporary)): digest(path)
        for path in temporary.rglob("*")
        if path.is_file()
    }
    (temporary / "reproducibility_manifest.json").write_text(
        json.dumps({"contract": contract, "artifact_sha256": hashes}, indent=2) + "\n"
    )
    temporary.rename(final)
    for name, expected in hashes.items():
        if digest(final / name) != expected:
            raise ValueError(f"Drive final artifact failed verification: {name}")


def evaluate_one(condition: str) -> None:
    """Worker process: existing inference backend, fresh base and one adapter."""
    from dataclasses import replace

    from multilingual_llm_reasoning.baseline import evaluate_baseline
    from multilingual_llm_reasoning.comparison_backends import AdapterBackend

    config, _ = preflight()
    adapter_path = (
        config.english_adapter if condition == "english_lora" else config.multilingual_adapter
    )
    inputs, _, _ = load_baseline_inputs(config.baseline)
    test_ids = {row["original_problem_id"] for row in inputs}
    audit = audit_adapter(adapter_path, condition, config.baseline, test_ids)
    baseline = replace(config.baseline, output_dir=config.baseline.output_dir / condition)

    def factory(cfg, revision):
        backend = AdapterBackend(cfg, revision, condition, adapter_path)
        backend.metadata["adapter_audit"] = audit["files"]
        return backend

    result = evaluate_baseline(
        baseline, allow_full_run=True, condition_id=condition, backend_factory=factory
    )
    if not result["complete"] or result["expected_variants"] != 392:
        raise ValueError("Adapter evaluation incomplete; inspect inference_errors.jsonl")


def mark_condition_complete(local_root: Path, drive_root: Path, condition: str) -> None:
    local = local_root / condition
    manifest = json.loads((local / "manifest.json").read_text())
    if not manifest.get("complete") or manifest.get("expected_variants") != 392:
        raise ValueError("Condition did not complete all 392 variants")
    target = drive_root / "progress" / condition
    marker = target / "completed.json"
    if marker.exists():
        raise FileExistsError("Condition completion marker already exists")
    artifacts = {}
    for name in ("manifest.json", "predictions.jsonl"):
        source = local / name
        shutil.copy2(source, target / name)
        if digest(source) != digest(target / name):
            raise ValueError(f"Drive condition artifact copy failed: {name}")
        artifacts[name] = digest(source)
    marker.write_text(
        json.dumps(
            {"status": "completed", "n_variants": 392, "artifacts_sha256": artifacts}, indent=2
        )
        + "\n"
    )


def run(stage: str, drive_root: Path, *, resume: bool, interval: float) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Phase 16L adapter evaluation requires an NVIDIA CUDA GPU")
    if str(drive_root).startswith("/content/drive/") and not Path("/content/drive").is_mount():
        raise ValueError("Google Drive is not mounted")
    config, contract = preflight()
    local_root = config.baseline.output_dir
    initialize_destination(drive_root, contract, resume=resume)
    if resume:
        restore_progress(local_root, drive_root)
    if stage == "finalize":
        from multilingual_llm_reasoning.comparison import evaluate_comparison

        for condition in CONDITIONS:
            marker = drive_root / "progress" / condition / "completed.json"
            if not marker.is_file():
                raise ValueError(f"Condition is not complete: {condition}")
        report = evaluate_comparison(config, allow_full_run=True)
        if not report["complete"] or report["expected_predictions"] != 1176:
            raise ValueError("Final comparison incomplete")
        persist_final(local_root, drive_root, contract)
        print(
            json.dumps(
                {
                    "status": "completed",
                    "research_ready": False,
                    "expected_predictions": 1176,
                    "drive": str(drive_root / "final"),
                },
                indent=2,
            )
        )
        return
    marker = drive_root / "progress" / stage / "completed.json"
    if marker.exists():
        raise FileExistsError(f"Condition already completed: {stage}")
    local = local_root / stage
    if not resume and local.exists() and any(local.iterdir()):
        raise FileExistsError("Local condition output already exists; use explicit resume")
    command = [sys.executable, str(Path(__file__).resolve()), "--worker-condition", stage]
    child = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
    try:
        while child.poll() is None:
            persist_progress(local_root, drive_root)
            time.sleep(interval)
        progress = persist_progress(local_root, drive_root)
    except Exception:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=30)
        raise
    if child.returncode != 0:
        raise RuntimeError(f"Adapter evaluation exited {child.returncode}; progress: {progress}")
    mark_condition_complete(local_root, drive_root, stage)
    print(json.dumps({"condition": stage, "status": "completed", "n_variants": 392}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", type=Path)
    parser.add_argument("--stage", choices=(*CONDITIONS, "finalize"))
    parser.add_argument("--worker-condition", choices=CONDITIONS)
    parser.add_argument("--resume-from-drive", action="store_true")
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--confirm-evaluation", action="store_true")
    args = parser.parse_args()
    if args.worker_condition is not None:
        evaluate_one(args.worker_condition)
        return
    if args.drive_root is None or args.stage is None:
        parser.error("--drive-root and --stage are required")
    if not args.confirm_evaluation:
        raise SystemExit("Explicit --confirm-evaluation is required")
    if not 0 < args.interval_seconds <= 300:
        raise SystemExit("Backup interval must be in (0, 300] seconds")
    run(
        args.stage,
        args.drive_root.resolve(),
        resume=args.resume_from_drive,
        interval=args.interval_seconds,
    )


if __name__ == "__main__":
    main()
