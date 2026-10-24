"""Three-condition evaluation on one immutable, held-out multilingual input grid."""

import gc
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import yaml
from filelock import FileLock

from .baseline import _hash, _json, _jsonl, _write, evaluate_baseline, load_baseline_inputs
from .baseline_backends import MockBaselineBackend
from .baseline_config import (
    BaselineConfig,
    baseline_config_dict,
    decoding_settings,
    load_baseline_config,
    make_messages,
)
from .baseline_scoring import score_record
from .comparison_backends import AdapterBackend, audit_adapter
from .comparison_metrics import CONDITIONS, export_summary, paired_summary
from .translation import LANGUAGES


@dataclass(frozen=True)
class ComparisonConfig:
    baseline: BaselineConfig
    english_adapter: Path | None
    multilingual_adapter: Path | None
    bootstrap_seed: int
    bootstrap_samples: int
    allow_provisional: bool = False
    baseline_reuse_manifest: Path | None = None


def load_comparison_config(path: Path) -> ComparisonConfig:
    raw = yaml.safe_load(path.read_text())
    if (
        not isinstance(raw, dict)
        or not {
            "schema_version",
            "baseline_config",
            "english_adapter",
            "multilingual_adapter",
            "bootstrap_seed",
            "bootstrap_samples",
        }
        <= set(raw)
        or set(raw)
        - {
            "schema_version",
            "baseline_config",
            "english_adapter",
            "multilingual_adapter",
            "bootstrap_seed",
            "bootstrap_samples",
            "allow_provisional",
            "baseline_reuse_manifest",
        }
        or type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
    ):
        raise ValueError("Invalid comparison config schema")
    base = load_baseline_config((path.parent / raw["baseline_config"]).resolve())
    if base.languages != LANGUAGES:
        raise ValueError("Three-condition comparison requires all four languages")
    if base.split != "test" or base.backend not in ("mock", "transformers"):
        raise ValueError("Comparison requires official test and mock/transformers backend")
    if base.backend != "mock" and (
        (base.allow_unreviewed_translations and raw.get("allow_provisional") is not True)
        or not re.fullmatch(r"[a-f0-9]{40}", base.revision)
    ):
        raise ValueError(
            "Real comparisons require reviewed translations or explicit provisional opt-in "
            "and a pinned base revision"
        )
    if "allow_provisional" in raw and type(raw["allow_provisional"]) is not bool:
        raise ValueError("allow_provisional must be boolean")
    for key in ("bootstrap_seed", "bootstrap_samples"):
        if type(raw[key]) is not int or raw[key] < (100 if key == "bootstrap_samples" else 0):
            raise ValueError("Invalid bootstrap settings")
    paths = [
        None if raw[k] is None else (path.parent / raw[k]).resolve()
        for k in ("english_adapter", "multilingual_adapter")
    ]
    if base.backend != "mock" and (None in paths or paths[0] == paths[1]):
        raise ValueError("Supply two distinct adapter paths for real evaluation")
    reuse = raw.get("baseline_reuse_manifest")
    if reuse is not None and (not isinstance(reuse, str) or not reuse.strip()):
        raise ValueError("baseline_reuse_manifest must be a path or null")
    reuse_path = (path.parent / reuse).resolve() if reuse else None
    return ComparisonConfig(
        base,
        *paths,
        raw["bootstrap_seed"],
        raw["bootstrap_samples"],
        raw.get("allow_provisional", False),
        reuse_path,
    )


def verify_baseline_reuse(
    path: Path, config: BaselineConfig, inputs: list[dict], source_hash: str
) -> tuple[list[dict], dict]:
    """Reuse raw baseline responses only after a complete variant/protocol audit."""
    manifest = json.loads(path.read_text())
    old = manifest["config"]
    if (
        not manifest.get("complete")
        or manifest.get("mock_model")
        or manifest.get("translation_manifest_sha256") != source_hash
        or manifest.get("expected_variants") != len(inputs)
    ):
        raise ValueError("Saved baseline is incomplete, mock, or from different translations")
    for field in (
        "backend",
        "model",
        "revision",
        "dtype",
        "batch_size",
        "max_new_tokens",
        "temperature",
        "do_sample",
        "seed",
        "num_questions",
        "split",
        "prompt_version",
    ):
        if old.get(field) != baseline_config_dict(config).get(field):
            raise ValueError(f"Saved baseline protocol differs: {field}")
    if old.get("languages", list(LANGUAGES)) != list(config.languages):
        raise ValueError("Saved baseline language scope differs")
    artifact = manifest["artifacts"]["predictions"]
    predictions_path = (path.parent / artifact["path"]).resolve()
    if not predictions_path.is_relative_to(path.parent.resolve()):
        raise ValueError("Saved baseline prediction path escapes its manifest directory")
    content = predictions_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != artifact["sha256"]:
        raise ValueError("Saved baseline prediction hash mismatch")
    rows = [json.loads(line) for line in content.splitlines()]
    expected = {(row["original_problem_id"], row["language"]): row for row in inputs}
    if len(rows) != len(expected) or len(rows) != artifact["count"]:
        raise ValueError("Saved baseline prediction count differs")
    seen = set()
    for row in rows:
        key = (row.get("original_problem_id"), row.get("language"))
        if key in seen or key not in expected:
            raise ValueError("Saved baseline has duplicate or unexpected variants")
        seen.add(key)
        source = expected[key]
        if (
            row.get("inference_status") != "completed"
            or row.get("backend") != "transformers"
            or row.get("model_checkpoint") != config.model
            or row.get("model_revision") != config.revision
            or row.get("prompt_template_version") != config.prompt_version
            or row.get("prompt_messages") != make_messages(source["translated_text"])
            or row.get("input_translation_sha256") != _hash(source["translated_text"])
            or row.get("decoding") != decoding_settings(config)
            or row.get("gold_answer") != source["gold_answer"]
            or row.get("parser_version") != score_record(row)["parser_version"]
        ):
            raise ValueError(f"Saved baseline variant/protocol mismatch: {key}")
    if set(expected) != seen:
        raise ValueError("Saved baseline is missing variants")
    backend = manifest.get("backend") or {}
    if (
        backend.get("model_checkpoint") != config.model
        or backend.get("model_revision") != config.revision
        or backend.get("dtype") != config.dtype
    ):
        raise ValueError("Saved baseline backend metadata differs")
    return rows, manifest


def evaluate_comparison(
    config: ComparisonConfig, *, allow_full_run: bool = False, retry_errors: bool = False
) -> dict:
    base = config.baseline
    if base.languages != LANGUAGES:
        raise ValueError("Three-condition comparison requires all four languages")
    if base.num_questions > 5 and not allow_full_run:
        raise ValueError("Start with five originals; more requires --allow-full-run")
    inputs, source_hash, reviewed = load_baseline_inputs(base)
    if not reviewed and not config.allow_provisional:
        raise ValueError("Provisional translations require explicit comparison opt-in")
    test_ids = {r["original_problem_id"] for r in inputs}
    paths = {
        "base": None,
        "english_lora": config.english_adapter,
        "multilingual_lora": config.multilingual_adapter,
    }
    adapters = (
        {}
        if base.backend == "mock"
        else {c: audit_adapter(paths[c], c, base, test_ids) for c in CONDITIONS if c != "base"}
    )
    if adapters:
        a, b = [adapters[c]["provenance"] for c in CONDITIONS[1:]]
        for key in ("train_ids", "validation_ids", "training_budget"):
            if a[key] != b[key]:
                raise ValueError(f"Adapter conditions are not matched: {key}")
    contract_config = asdict(config)
    contract_config["baseline"] = baseline_config_dict(base)
    contract = {
        "config": contract_config,
        "translation_manifest_sha256": source_hash,
        "selected_original_ids": sorted(test_ids),
        "adapters": adapters,
        "comparison_version": 1,
    }
    root = base.output_dir
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".comparison.lock"), timeout=0):
        signature = _hash(_json(contract))
        contract_path = root / "comparison_contract.json"
        if (
            contract_path.exists()
            and json.loads(contract_path.read_text())["signature"] != signature
        ):
            raise ValueError(
                "Comparison data/config/adapters changed; choose a new output directory"
            )
        _write(contract_path, _json({"signature": signature, **contract}) + "\n")
        _jsonl(root / "selected_test.jsonl", inputs)
        records, manifests = [], {}
        for condition in CONDITIONS:
            condition_config = replace(base, output_dir=root / condition)

            if condition == "base" and config.baseline_reuse_manifest is not None:
                reused, source_manifest = verify_baseline_reuse(
                    config.baseline_reuse_manifest, condition_config, inputs, source_hash
                )
                condition_config.output_dir.mkdir(parents=True, exist_ok=True)
                _jsonl(
                    condition_config.output_dir / "predictions.jsonl",
                    [{**row, "condition_id": "base"} for row in reused],
                )
                manifests[condition] = {
                    **source_manifest,
                    "baseline_reused_from": str(config.baseline_reuse_manifest),
                }
                records.extend({**row, "condition_id": "base"} for row in reused)
                continue

            def factory(cfg, revision, condition=condition):
                if cfg.backend == "mock":
                    backend = MockBaselineBackend(cfg, revision)
                    backend.metadata.update(
                        condition_id=condition, adapter_activation={"mock": True}
                    )
                else:
                    backend = AdapterBackend(cfg, revision, condition, paths[condition])
                backend.metadata["adapter_audit"] = adapters.get(condition, {}).get("files")
                return backend

            result = evaluate_baseline(
                condition_config,
                allow_full_run=allow_full_run,
                retry_errors=retry_errors,
                backend_factory=factory,
                condition_id=condition,
            )
            if (
                result["translation_manifest_sha256"] != source_hash
                or set(result["selected_original_ids"]) != test_ids
            ):
                raise ValueError(
                    "Test data changed between conditions; preserve inputs "
                    "and use a new output directory"
                )
            manifests[condition] = result
            for line in (
                (condition_config.output_dir / "predictions.jsonl").read_bytes().splitlines()
            ):
                records.append({**json.loads(line), "condition_id": condition})
            gc.collect()
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        # Successful runs must share resolved precision, checkpoint and exact chat template.
        metadata = [m["backend"] for m in manifests.values() if m.get("backend")]
        for field in (
            "model_checkpoint",
            "model_revision",
            "chat_template_sha256",
            "dtype",
            *(("device",) if config.baseline_reuse_manifest is None else ()),
        ):
            if len({_json(m[field]) for m in metadata}) > 1:
                raise ValueError(f"Condition inference settings differ: {field}")
        predictions = _jsonl(root / "predictions.jsonl", records)
        report = paired_summary(
            records, inputs, seed=config.bootstrap_seed, bootstrap_samples=config.bootstrap_samples
        )
        report.update(
            mock=base.backend == "mock",
            research_ready=reviewed
            and base.backend != "mock"
            and all(m["complete"] for m in manifests.values()),
            complete=all(m["complete"] for m in manifests.values()),
            comparison_signature=signature,
            predictions=predictions,
            interpretation=(
                "No improvement claim is made. "
                "Mock, incomplete, or provisional-translation results require caution."
            ),
            baseline_reused=config.baseline_reuse_manifest is not None,
            baseline_device_difference=(
                "Reused baseline was generated on a different device; model, dtype, "
                "prompt and decoding were audited, but cross-device numerics can differ."
                if config.baseline_reuse_manifest is not None
                else None
            ),
        )
        export_summary(report, root)
        return report
