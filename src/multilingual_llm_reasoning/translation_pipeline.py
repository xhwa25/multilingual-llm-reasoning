"""Translation orchestration. Raw cached responses are never modified by review."""

import csv
import hashlib
import io
import json
import random
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

from .logging import configure_logging
from .preparation import assert_split_integrity, original_problem_id
from .translation import (
    ADAPTER_VERSION,
    LANGUAGES,
    PROMPT_VERSION,
    LocalTransformersTranslationProvider,
    MockTranslationProvider,
    OpenAITranslationProvider,
    TranslationError,
    TranslationProvider,
    TranslationResult,
    translation_prompt,
)
from .translation_config import TranslationConfig
from .translation_qa import qa_flags
from .translation_store import LimitReached, TranslationStore, utc_now

Record = dict[str, Any]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def load_translation_inputs(config: TranslationConfig) -> tuple[list[Record], str]:
    manifest_bytes = config.manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("status") != "ok" or manifest.get("schema_version") != 1:
        raise ValueError("Translation requires a successful preparation manifest")
    root = config.manifest_path.parent

    def read(artifact: Record) -> list[Record]:
        path = (root / artifact["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Dataset artifact escapes manifest directory")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != artifact["sha256"]:
            raise ValueError(f"Dataset artifact hash mismatch: {artifact['path']}")
        # JSON strings may contain U+2028/U+2029; only LF separates JSONL records.
        rows = [json.loads(line) for line in content.split(b"\n") if line]
        if len(rows) != artifact["count"]:
            raise ValueError("Dataset artifact count mismatch")
        return rows

    canonical = {
        split: read(manifest["splits"][split]) for split in ("train", "validation", "test")
    }
    assert_split_integrity(canonical)
    selected = {}
    for split in config.splits:
        rows = canonical[split] if split == "validation" else read(manifest["subsets"][split])
        canonical_by_id = {r["original_problem_id"]: r for r in canonical[split]}
        for row in rows:
            identity = row["original_problem_id"]
            if canonical_by_id.get(identity) != row:
                raise ValueError("Subset record differs from its canonical split")
            if identity != original_problem_id(row["original_question"]):
                raise ValueError("Original ID does not match source question")
        selected[split] = rows
    assert_split_integrity(selected)
    if not any(selected.values()):
        raise ValueError("No questions selected")
    return [row for rows in selected.values() for row in rows], hashlib.sha256(
        manifest_bytes
    ).hexdigest()


def _signature(config: TranslationConfig, manifest_hash: str) -> str:
    return _digest(
        _json(
            {
                "manifest": manifest_hash,
                "splits": config.splits,
                "provider": config.provider,
                "model": config.model,
                "adapter_version": ADAPTER_VERSION,
                "prompt_version": config.prompt_version,
                "prompts": {lang: translation_prompt(lang) for lang in LANGUAGES[1:]},
                "max_output_tokens": config.max_output_tokens,
                "input_price": config.input_usd_per_million,
                "output_price": config.output_usd_per_million,
            }
        )
    )


def cache_key(config: TranslationConfig, row: Record, language: str) -> str:
    return _digest(
        _json(
            {
                "id": row["original_problem_id"],
                "source_text": row["original_question"],
                "language": language,
                "provider": config.provider,
                "model": config.model,
                "provider_version": ADAPTER_VERSION,
                "prompt_version": config.prompt_version,
                "instructions": translation_prompt(language) if language != "en" else "identity-v1",
                "max_output_tokens": config.max_output_tokens,
            }
        )
    )


def _input_allowance(text: str, language: str) -> int:
    # Conservative text-only byte allowance plus protocol overhead, NOT exact tokenization.
    return len((translation_prompt(language) + text).encode("utf-8")) + 1024


def _cost(config: TranslationConfig, input_tokens: int, output_tokens: int) -> Decimal:
    if config.provider in ("mock", "local_transformers"):
        return Decimal(0)
    if config.input_usd_per_million is None or config.output_usd_per_million is None:
        raise ValueError("Set both per-million token prices from the current provider price page")
    return (
        input_tokens * config.input_usd_per_million + output_tokens * config.output_usd_per_million
    ) / 1_000_000


def estimate_translation(config: TranslationConfig) -> Record:
    """Offline estimate: no client creation, credential access or network requests."""
    rows, manifest_hash = load_translation_inputs(config)
    input_tokens = sum(
        _input_allowance(row["original_question"], lang) for row in rows for lang in LANGUAGES[1:]
    )
    requests = len(rows) * 3
    priced = config.provider in ("mock", "local_transformers") or (
        config.input_usd_per_million is not None and config.output_usd_per_million is not None
    )
    cost = _cost(config, input_tokens, requests * config.max_output_tokens) if priced else None
    return {
        "provider": config.provider,
        "model": config.model,
        "selected_questions": len(rows),
        "variants": len(rows) * 4,
        "uncached_requests": requests,
        "max_attempts_per_translation": config.max_attempts,
        "input_token_allowance": input_tokens,
        "maximum_output_tokens_first_attempts": requests * config.max_output_tokens,
        "first_attempt_allowance_usd": str(cost) if cost is not None else None,
        "all_retries_allowance_usd": str(cost * config.max_attempts) if cost is not None else None,
        "max_requests": config.max_requests,
        "max_cost_usd": str(config.max_cost_usd),
        "manifest_sha256": manifest_hash,
        "note": "Cold-cache conservative byte-based allowance, not a quote or exact token count. "
        "Retries count toward limits. Resume uses the persistent ledger; cached results are free.",
    }


def _raw_record(
    config: TranslationConfig, row: Record, lang: str, key: str, result: TranslationResult
) -> Record:
    identity = lang == "en"
    flags = (
        []
        if identity
        else qa_flags(
            row["original_question"],
            result.text,
            lang,
            gold_answer=row["gold_answer"],
            mock=config.provider == "mock",
            response_status=result.status,
        )
    )
    return {
        "original_problem_id": row["original_problem_id"],
        "split": row["split"],
        "source_split": row["source_split"],
        "source_text": row["original_question"],
        "translated_text": result.text,
        "language": lang,
        "translation_provider": "identity" if identity else config.provider,
        "translation_model": "original" if identity else config.model,
        "translation_model_version": result.model,
        "translation_provider_version": ADAPTER_VERSION,
        "prompt_version": "identity-v1"
        if identity
        else ("nllb-direct-v1" if config.provider == "local_transformers" else PROMPT_VERSION),
        "prompt_sha256": (
            None
            if identity or config.provider == "local_transformers"
            else _digest(translation_prompt(lang))
        ),
        "timestamp": utc_now(),
        "cache_key": key,
        "raw_translation_sha256": _digest(result.text),
        "response_id": result.response_id,
        "response_status": result.status,
        "usage": {"input_tokens": result.input_tokens, "output_tokens": result.output_tokens},
        "qa_status": "source_original" if identity else ("flagged" if flags else "needs_review"),
        "qa_flags": flags,
    }


def _translate_one(
    config: TranslationConfig,
    store: TranslationStore,
    provider: TranslationProvider,
    row: Record,
    lang: str,
    key: str,
    sleep: Callable[[float], None],
) -> Record | None:
    prior = store.previous_attempts(key)
    if any(state == "permanent" for state, _ in prior):
        return None
    for attempt_number in range(len(prior), config.max_attempts):
        reservation = _cost(
            config, _input_allowance(row["original_question"], lang), config.max_output_tokens
        )
        attempt = store.reserve(
            key, reservation, max_requests=config.max_requests, max_cost=config.max_cost_usd
        )
        if attempt_number:
            base = min(
                config.max_backoff_seconds, config.backoff_seconds * 2 ** (attempt_number - 1)
            )
            jitter = random.Random(f"{config.seed}:{key}:{attempt_number}").uniform(0.5, 1.0)
            sleep(base * jitter)
        try:
            result = provider.translate(
                row["original_question"],
                language=lang,
                instructions=translation_prompt(lang),
                max_output_tokens=config.max_output_tokens,
            )
            if (
                not isinstance(result.text, str)
                or type(result.input_tokens) is not int
                or result.input_tokens < 0
                or type(result.output_tokens) is not int
                or result.output_tokens < 0
            ):
                raise TranslationError("invalid_provider_result")
        except TranslationError as exc:
            store.failure(attempt, str(exc), retryable=exc.retryable)
            if not exc.retryable:
                return None
            continue
        record = _raw_record(config, row, lang, key, result)
        actual = _cost(config, result.input_tokens, result.output_tokens)
        store.success(attempt, key, record, actual)
        if actual > reservation:
            raise LimitReached("Actual usage exceeded estimated reservation; inspect prices/usage")
        return record
    return None


def _import_corrections(path: Path, store: TranslationStore, raw: list[Record]) -> None:
    by_pair = {(r["original_problem_id"], r["language"]): r for r in raw}
    events = []
    fields = {
        "original_problem_id",
        "language",
        "raw_translation_sha256",
        "corrected_text",
        "decision",
        "reviewer",
        "reviewed_at",
        "notes",
    }
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not fields <= set(reader.fieldnames or []):
                raise ValueError("Review CSV is missing required review columns")
            imported = [
                {key: (row[key] or None) if key == "corrected_text" else row[key] for key in fields}
                for row in reader
                if row["decision"].strip()
            ]
    else:
        imported = [json.loads(line) for line in path.read_bytes().split(b"\n") if line]
    for index, event in enumerate(imported, 1):
        if not isinstance(event, dict) or set(event) != fields:
            raise ValueError(f"Correction line {index}: requires exactly {sorted(fields)}")
        if any(
            not isinstance(event[key], str)
            for key in ("original_problem_id", "language", "raw_translation_sha256", "decision")
        ):
            raise ValueError("Review identity, hash and decision must be strings")
        pair = (event["original_problem_id"], event["language"])
        original = by_pair.get(pair)
        if original is None or event["language"] == "en":
            raise ValueError(f"Correction line {index}: unknown variant or immutable English")
        if event["raw_translation_sha256"] != original["raw_translation_sha256"]:
            raise ValueError(f"Correction line {index}: stale raw translation hash")
        if event["decision"] not in ("approved", "rejected"):
            raise ValueError("Review decision must be approved or rejected")
        for field in ("reviewer", "reviewed_at", "notes"):
            if not isinstance(event[field], str) or not event[field].strip():
                raise ValueError(f"Review requires nonempty {field}")
        if datetime.fromisoformat(event["reviewed_at"]).utcoffset() is None:
            raise ValueError("reviewed_at must include a timezone")
        corrected = event["corrected_text"]
        if corrected is not None and (not isinstance(corrected, str) or not corrected.strip()):
            raise ValueError("corrected_text must be null or a nonempty string")
        events.append((_digest(_json(event)), event))
    # Validate the entire file before appending any events; re-import is idempotent.
    store.import_reviews(events)


def _atomic_write(path: Path, content: str) -> Record:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
    return {"path": path.name, "sha256": _digest(content)}


def _jsonl(path: Path, rows: list[Record]) -> Record:
    return {**_atomic_write(path, "".join(_json(row) + "\n" for row in rows)), "count": len(rows)}


def _csv_safe(value: object) -> str:
    text = str(value)
    # Do not allow model/source text to execute formulas when opened in a spreadsheet.
    return "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) else text


def _export(
    config: TranslationConfig,
    store: TranslationStore,
    rows: list[Record],
    raw: list[Record],
    manifest_hash: str,
    blocked: str | None,
) -> Record:
    root = config.output_dir
    raw_by_pair = {(r["original_problem_id"], r["language"]): r for r in raw}
    reviews = store.reviews()
    latest = {(e["original_problem_id"], e["language"]): e for e in reviews}
    effective = []
    for row in rows:
        for lang in LANGUAGES:
            pair = (row["original_problem_id"], lang)
            original = raw_by_pair.get(pair)
            if original is None:
                key = cache_key(config, row, lang)
                attempts = store.previous_attempts(key)
                effective.append(
                    {
                        "original_problem_id": pair[0],
                        "language": lang,
                        "split": row["split"],
                        "source_split": row["source_split"],
                        "source_text": row["original_question"],
                        "translated_text": None,
                        "qa_status": "translation_error",
                        "qa_flags": ["missing_translation"],
                        "cache_key": key,
                        "translation_provider": config.provider,
                        "translation_model": config.model,
                        "translation_provider_version": ADAPTER_VERSION,
                        "prompt_version": config.prompt_version,
                        "error": attempts[-1][1] if attempts else "not_requested",
                    }
                )
                continue
            result = {**original, "gold_answer": row["gold_answer"], "review": None}
            if pair in latest:
                review = latest[pair]
                result["translated_text"] = review["corrected_text"] or original["translated_text"]
                result["qa_flags"] = qa_flags(
                    row["original_question"],
                    result["translated_text"],
                    lang,
                    mock=config.provider == "mock",
                )
                result["qa_status"] = "human_" + review["decision"]
                result["review"] = review
            effective.append(result)
    ranked_ids = sorted(
        {row["original_problem_id"] for row in rows},
        key=lambda identity: _digest(f"{config.seed}:review:{identity}"),
    )
    sampled = set(ranked_ids[: config.review_sample_size])
    review_rows = [r for r in effective if r["original_problem_id"] in sampled or r["qa_flags"]]
    buffer = io.StringIO(newline="")
    fields = [
        "original_problem_id",
        "split",
        "language",
        "source_text",
        "raw_translated_text",
        "translated_text",
        "raw_translation_sha256",
        "qa_status",
        "qa_flags",
        "sampled",
        "reviewer",
        "decision",
        "corrected_text",
        "reviewed_at",
        "notes",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in review_rows:
        original = raw_by_pair.get((row["original_problem_id"], row["language"]), {})
        out = {key: row.get(key, "") for key in fields}
        out.update(
            raw_translated_text=original.get("translated_text", ""),
            sampled=row["original_problem_id"] in sampled,
            qa_flags=";".join(row["qa_flags"]),
        )
        writer.writerow({key: _csv_safe(value) for key, value in out.items()})
    attempts, charged = store.totals()
    report = {
        "schema_version": 1,
        "input_manifest_sha256": manifest_hash,
        "provider": config.provider,
        "model": config.model,
        "prompt_version": config.prompt_version,
        "languages": LANGUAGES,
        "selected_questions": len(rows),
        "expected_variants": len(rows) * 4,
        "cached_variants": len(raw),
        "complete": len(raw) == len(rows) * 4,
        "mock": config.provider == "mock",
        "blocked_reason": blocked,
        "request_attempts": attempts,
        "accounted_cost_usd": str(charged),
        "limits": {"max_requests": config.max_requests, "max_cost_usd": str(config.max_cost_usd)},
        "qa_counts": dict(Counter(r["qa_status"] for r in effective)),
        "flag_counts": dict(Counter(flag for row in effective for flag in row["qa_flags"])),
        "variants_by_language": dict(Counter(row["language"] for row in effective)),
        "review_sample_original_questions": len(sampled),
        "meaning_preservation_automatically_verified": False,
        "research_ready": config.provider != "mock"
        and all(
            r["qa_status"] in ("source_original", "human_approved") and r["translated_text"]
            for r in effective
        ),
        "splits": dict(Counter(row["split"] for row in rows)),
        "artifacts": {
            "raw": _jsonl(root / "raw_translations.jsonl", raw),
            "variants": _jsonl(root / "translations.jsonl", effective),
            "reviews": _jsonl(root / "review_events.jsonl", reviews),
            "attempts": _jsonl(root / "request_ledger.jsonl", store.attempt_records()),
            "human_review": {
                **_atomic_write(root / "human_review.csv", buffer.getvalue()),
                "count": len(review_rows),
            },
        },
    }
    _atomic_write(root / "qa_report.json", _json(report) + "\n")
    # Manifest is written last; it authenticates all exports, including QA.
    _atomic_write(
        root / "manifest.json",
        _json(
            {
                **report,
                "config": asdict(config),
                "qa_report_sha256": hashlib.sha256(
                    (root / "qa_report.json").read_bytes()
                ).hexdigest(),
            }
        )
        + "\n",
    )
    return report


def translate_dataset(
    config: TranslationConfig,
    *,
    allow_paid: bool = False,
    corrections: Path | None = None,
    review_only: bool = False,
    provider: TranslationProvider | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Record:
    rows, manifest_hash = load_translation_inputs(config)
    if config.provider == "openai" and not review_only:
        if not allow_paid:
            raise ValueError(
                "API requests disabled. Run --estimate-only first; --allow-paid is required"
            )
        if config.model.startswith("REPLACE_"):
            raise ValueError("Set model to your chosen available model snapshot")
        if any(
            price is None or price <= 0
            for price in (config.input_usd_per_million, config.output_usd_per_million)
        ):
            raise ValueError("Set positive current input/output prices before paid requests")
    if review_only and not (config.output_dir / "translations.sqlite3").exists():
        raise ValueError("--review-only requires an existing translation run")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(config.output_dir / ".translation.lock"), timeout=0):
            store = TranslationStore(
                config.output_dir / "translations.sqlite3", _signature(config, manifest_hash)
            )
            try:
                blocked = None
                for row in rows:
                    for lang in LANGUAGES:
                        key = cache_key(config, row, lang)
                        if store.cached(key) or review_only:
                            continue
                        if lang == "en":
                            record = _raw_record(
                                config,
                                row,
                                lang,
                                key,
                                TranslationResult(row["original_question"], "original"),
                            )
                            store.put_identity(key, record)
                        elif not blocked:
                            if provider is None:
                                provider = (
                                    MockTranslationProvider()
                                    if config.provider == "mock"
                                    else LocalTransformersTranslationProvider()
                                    if config.provider == "local_transformers"
                                    else OpenAITranslationProvider(
                                        config.model, timeout_seconds=config.timeout_seconds
                                    )
                                )
                            try:
                                _translate_one(config, store, provider, row, lang, key, sleep)
                            except LimitReached as exc:
                                blocked = str(exc)
                raw = [
                    record
                    for row in rows
                    for lang in LANGUAGES
                    if (record := store.cached(cache_key(config, row, lang))) is not None
                ]
                if corrections is not None:
                    _import_corrections(corrections, store, raw)
                report = _export(config, store, rows, raw, manifest_hash, blocked)
                configure_logging().info("translation_export_completed")
                return report
            finally:
                store.close()
    except Timeout as exc:
        raise ValueError("Another translation process is using this output directory") from exc
