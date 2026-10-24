"""Resumable batched baseline evaluation with one exported row per selected variant."""

import hashlib
import json
import logging
import sqlite3
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

from .baseline_backends import BaselineBackend, Generation, create_backend, resolve_revision
from .baseline_config import (
    INSTRUCTION,
    BaselineConfig,
    baseline_config_dict,
    decoding_settings,
    make_messages,
)
from .baseline_scoring import score_record, summarize_scores
from .logging import configure_logging
from .numeric import normalize_numeric
from .preparation import original_problem_id
from .translation import LANGUAGES

Record = dict[str, Any]


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _write(path: Path, text: str) -> Record:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return {"path": path.name, "sha256": _hash(text)}


def _jsonl(path: Path, rows: list[Record]) -> Record:
    return {**_write(path, "".join(_json(row) + "\n" for row in rows)), "count": len(rows)}


def load_baseline_inputs(config: BaselineConfig) -> tuple[list[Record], str, bool]:
    if config.languages not in (("en",), LANGUAGES):
        raise ValueError("Unsupported evaluation language scope")
    manifest_bytes = config.translation_manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != 1 or not manifest.get("complete"):
        raise ValueError("Baseline requires a complete translation manifest")
    root = config.translation_manifest.parent.resolve()
    artifact = manifest["artifacts"]["variants"]
    path = (root / artifact["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Translation artifact escapes manifest directory")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != artifact["sha256"]:
        raise ValueError("Translation artifact hash mismatch")
    rows = [json.loads(line) for line in content.splitlines()]
    if len(rows) != artifact["count"]:
        raise ValueError("Translation record count mismatch")
    groups: dict[str, dict[str, Record]] = defaultdict(dict)
    for row in rows:
        identity, language = row["original_problem_id"], row["language"]
        if language not in LANGUAGES or language in groups[identity]:
            raise ValueError("Unknown language or duplicate original ID/language")
        if not isinstance(row.get("translated_text"), str) or not row["translated_text"].strip():
            raise ValueError("Missing translated text; resolve QA before inference")
        if original_problem_id(row["source_text"]) != identity:
            raise ValueError("Translation source text does not match original ID")
        normalize_numeric(row.get("gold_answer"))
        groups[identity][language] = row
    for group in groups.values():
        if set(group) != set(LANGUAGES):
            raise ValueError("Every original must have en, zh, de and ja variants")
        if len({r["split"] for r in group.values()}) != 1:
            raise ValueError("An original problem appears in multiple splits")
        if len({normalize_numeric(r["gold_answer"]) for r in group.values()}) != 1:
            raise ValueError("Gold answers differ across languages")
        for row in group.values():
            expected = "test" if row["split"] == "test" else "train"
            if row["source_split"] != expected:
                raise ValueError("Original source split changed")
        if group["en"]["translated_text"] != group["en"]["source_text"]:
            raise ValueError("English must remain the unmodified source question")
    eligible = [
        identity for identity, group in groups.items() if group["en"]["split"] == config.split
    ]
    if len(eligible) < config.num_questions:
        raise ValueError("Requested more distinct questions than available; no silent shrinking")
    selected = sorted(eligible, key=lambda identity: _hash(f"{config.seed}:baseline:{identity}"))[
        : config.num_questions
    ]
    result = [groups[identity][lang] for identity in selected for lang in config.languages]
    reviewed = not manifest.get("mock", False) and all(
        row.get("qa_status") in ("source_original", "human_approved") for row in result
    )
    if not reviewed and not config.allow_unreviewed_translations:
        raise ValueError(
            "Selected translations are mock/unreviewed; use reviewed data or explicit opt-in"
        )
    return result, hashlib.sha256(manifest_bytes).hexdigest(), reviewed


class BaselineStore:
    def __init__(self, path: Path, signature: str) -> None:
        self.db = sqlite3.connect(path)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY, variant_key TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS anomalies (id INTEGER PRIMARY KEY, payload TEXT NOT NULL);
        """)
        prior = self.meta("signature")
        if prior is not None and prior != signature:
            self.db.close()
            raise ValueError("Baseline inputs/settings changed; use a new output directory")
        self.set_meta("signature", signature)

    def meta(self, key: str) -> Any:
        row = self.db.execute("SELECT payload FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set_meta(self, key: str, value: Any) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", (key, _json(value)))

    def latest(self, key: str) -> Record | None:
        row = self.db.execute(
            "SELECT payload FROM attempts WHERE variant_key=? ORDER BY id DESC LIMIT 1", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def start(self, key: str, row: Record) -> int:
        with self.db:
            cursor = self.db.execute(
                "INSERT INTO attempts(variant_key,payload) VALUES (?,?)", (key, _json(row))
            )
            return int(cursor.lastrowid)

    def finish(self, attempt: int, row: Record) -> None:
        with self.db:
            self.db.execute("UPDATE attempts SET payload=? WHERE id=?", (_json(row), attempt))

    def anomaly(self, row: Record) -> None:
        with self.db:
            self.db.execute("INSERT INTO anomalies(payload) VALUES (?)", (_json(row),))

    def all_attempts(self) -> list[Record]:
        return [
            {"attempt_id": row[0], **json.loads(row[1])}
            for row in self.db.execute("SELECT id,payload FROM attempts ORDER BY id")
        ]


def _key(row: Record) -> str:
    return row["original_problem_id"] + ":" + row["language"]


def _pending(config: BaselineConfig, row: Record, metadata: Record) -> Record:
    return score_record(
        {
            "condition_id": metadata.get("condition_id"),
            "adapter_activation": metadata.get("adapter_activation"),
            "adapter_audit": metadata.get("adapter_audit"),
            "original_problem_id": row["original_problem_id"],
            "language": row["language"],
            "split": row["split"],
            "source_split": row["source_split"],
            "requested_model": config.model,
            "requested_revision": config.revision,
            "model_checkpoint": metadata.get("model_checkpoint", config.model),
            "model_revision": metadata.get("model_revision"),
            "backend": config.backend,
            "device": metadata.get("device"),
            "dtype": metadata.get("dtype"),
            "backend_versions": metadata.get("versions", {}),
            "prompt_template_version": config.prompt_version,
            "instruction": INSTRUCTION,
            "prompt_messages": make_messages(row["translated_text"]),
            "chat_template": metadata.get("chat_template"),
            "chat_template_sha256": metadata.get("chat_template_sha256"),
            "decoding": decoding_settings(config),
            "input_qa_status": row["qa_status"],
            "input_translation_sha256": _hash(row["translated_text"]),
            "gold_answer": row["gold_answer"],
            "prediction": None,
            "inference_status": "pending",
            "error": None,
            "latency_seconds": None,
            "amortized_latency_seconds": None,
            "latency_kind": "batch_wall_time_including_tokenization_excluding_model_load",
            "timestamp": datetime.now(UTC).isoformat(),
        }
    )


def _export(
    config: BaselineConfig,
    store: BaselineStore,
    inputs: list[Record],
    source_hash: str,
    reviewed: bool,
) -> Record:
    records = [
        score_record(store.latest(_key(row)) or _pending(config, row, store.meta("backend") or {}))
        for row in inputs
    ]
    if len({(r["original_problem_id"], r["language"]) for r in records}) != len(inputs):
        raise ValueError("Output variant integrity failed")
    metrics = summarize_scores(records)
    root = config.output_dir
    artifacts = {
        "predictions": _jsonl(root / "predictions.jsonl", records),
        "attempts": _jsonl(root / "inference_attempts.jsonl", store.all_attempts()),
        "parse_failures": _jsonl(
            root / "parse_failures.jsonl",
            [
                r
                for r in records
                if r["parse_status"] not in ("ok", "inference_error", "pending", "invalid_gold")
            ],
        ),
        "inference_errors": _jsonl(
            root / "inference_errors.jsonl",
            [
                r
                for r in records
                if r["parse_status"] in ("inference_error", "pending", "invalid_gold")
            ],
        ),
        "anomalies": _jsonl(
            root / "backend_anomalies.jsonl",
            [
                json.loads(row[0])
                for row in store.db.execute("SELECT payload FROM anomalies ORDER BY id")
            ],
        ),
        "metrics": _write(root / "metrics.json", _json(metrics) + "\n"),
    }
    report = {
        "schema_version": 1,
        "config": baseline_config_dict(config),
        "translation_manifest_sha256": source_hash,
        "selected_original_ids": list(dict.fromkeys(row["original_problem_id"] for row in inputs)),
        "expected_variants": len(inputs),
        "variants_by_language": dict(Counter(row["language"] for row in records)),
        "backend": store.meta("backend"),
        "resolved_revision": store.meta("resolved_revision"),
        "complete": all(row["inference_status"] == "completed" for row in records),
        "research_ready_inputs": reviewed,
        "mock_model": config.backend == "mock",
        "interpretation": "Plumbing only"
        if config.backend == "mock" or not reviewed
        else "English instruction and shared output format may affect language comparisons",
        "artifacts": artifacts,
        "metrics": metrics,
    }
    _write(root / "manifest.json", _json(report) + "\n")
    return report


def evaluate_baseline(
    config: BaselineConfig,
    *,
    allow_full_run: bool = False,
    retry_errors: bool = False,
    condition_id: str | None = None,
    backend_factory: Callable[[BaselineConfig, str], BaselineBackend] = create_backend,
    revision_resolver: Callable[[BaselineConfig], str] = resolve_revision,
) -> Record:
    if config.num_questions > 5 and not allow_full_run:
        raise ValueError(
            "More than 5 originals requires --allow-full-run; start with the smoke test"
        )
    inputs, source_hash, reviewed = load_baseline_inputs(config)
    signature = _hash(
        _json(
            {
                "config": baseline_config_dict(config),
                "input_hash": source_hash,
                "instruction": INSTRUCTION,
                "pipeline_version": 1,
                **({"condition_id": condition_id} if condition_id is not None else {}),
            }
        )
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging()
    try:
        with FileLock(str(config.output_dir / ".baseline.lock"), timeout=0):
            store = BaselineStore(config.output_dir / "baseline.sqlite3", signature)
            try:
                store.set_meta("condition_id", condition_id)
                todo = []
                for row in inputs:
                    cached = store.latest(_key(row))
                    if (
                        cached is None
                        or cached["inference_status"] == "pending"
                        or (retry_errors and cached["inference_status"] == "error")
                    ):
                        todo.append(row)
                logger.info(
                    "baseline_started",
                    extra={
                        "details": {
                            "selected_variants": len(inputs),
                            "cached_variants": len(inputs) - len(todo),
                            "pending_variants": len(todo),
                            "backend": config.backend,
                        }
                    },
                )
                if todo:
                    _evaluate_pending(
                        config, store, todo, backend_factory, revision_resolver, logger
                    )
                return _export(config, store, inputs, source_hash, reviewed)
            finally:
                store.db.close()
    except Timeout as exc:
        raise ValueError("Another baseline process is using this output directory") from exc


def _evaluate_pending(
    config: BaselineConfig,
    store: BaselineStore,
    todo: list[Record],
    factory: Callable[[BaselineConfig, str], BaselineBackend],
    resolver: Callable[[BaselineConfig], str],
    logger: logging.Logger,
) -> None:
    try:
        revision = store.meta("resolved_revision")
        if revision is None:
            revision = resolver(config)
            store.set_meta("resolved_revision", revision)
        backend = factory(config, revision)
    except Exception as exc:
        metadata = store.meta("backend") or {
            "model_revision": store.meta("resolved_revision"),
            "condition_id": store.meta("condition_id"),
        }
        for row in todo:
            pending = _pending(config, row, metadata)
            attempt = store.start(_key(row), pending)
            store.finish(
                attempt,
                score_record(
                    {
                        **pending,
                        "inference_status": "error",
                        "error": f"backend_initialization: {type(exc).__name__}: {exc}",
                    }
                ),
            )
        logger.error(
            "baseline_initialization_failed", extra={"details": {"error_type": type(exc).__name__}}
        )
        return
    previous = store.meta("backend")
    if previous is not None and previous != backend.metadata:
        raise ValueError(
            "Resolved model/template/device/library versions changed; use a new output_dir"
        )
    store.set_meta("backend", backend.metadata)
    for offset in range(0, len(todo), config.batch_size):
        batch = todo[offset : offset + config.batch_size]
        pending = [_pending(config, row, backend.metadata) for row in batch]
        attempts = [
            store.start(_key(row), record) for row, record in zip(batch, pending, strict=True)
        ]
        started = time.perf_counter()
        error = None
        try:
            outputs = backend.generate([record["prompt_messages"] for record in pending])
            if not isinstance(outputs, list):
                store.anomaly(
                    {
                        "error": "invalid_backend_batch",
                        "batch_attempts": attempts,
                        "returned_output": repr(outputs),
                    }
                )
                raise ValueError("backend must return a list of Generation records")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            outputs = []
        elapsed = time.perf_counter() - started
        if len(outputs) > len(batch):
            store.anomaly(
                {
                    "error": "extra_backend_outputs",
                    "batch_attempts": attempts,
                    "outputs": [
                        asdict(output)
                        if isinstance(output, Generation)
                        else {"unparsed_output": repr(output)}
                        for output in outputs[len(batch) :]
                    ],
                }
            )
        for index, (attempt, record) in enumerate(zip(attempts, pending, strict=True)):
            output = outputs[index] if index < len(outputs) else None
            valid = isinstance(output, Generation) and isinstance(output.prediction, str)
            if not valid and output is not None:
                store.anomaly(
                    {
                        "error": "invalid_backend_output",
                        "attempt_id": attempt,
                        "output": asdict(output)
                        if isinstance(output, Generation)
                        else {"unparsed_output": repr(output)},
                    }
                )
            finalized = {
                **record,
                "prediction": output.prediction if valid else None,
                "inference_status": "completed" if valid else "error",
                "error": None if valid else error or "backend_output_missing_or_invalid",
                "finish_reason": output.finish_reason if valid else None,
                "generated_tokens": output.generated_tokens if valid else None,
                "latency_seconds": elapsed,
                "amortized_latency_seconds": elapsed / len(batch),
                "batch_size_actual": len(batch),
                "batch_attempt_ids": attempts,
            }
            store.finish(attempt, score_record(finalized))
        logger.info(
            "baseline_batch_completed",
            extra={
                "details": {
                    "processed_variants": min(offset + len(batch), len(todo)),
                    "total_pending": len(todo),
                    "batch_size": len(batch),
                    "batch_latency_seconds": elapsed,
                }
            },
        )
