"""Versioned GSM8K preparation: canonical splits, subset views and audit artifacts."""

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .logging import configure_logging
from .numeric import parse_gsm8k_answer

DATASET = "openai/gsm8k"
CONFIGURATION = "main"
Record = dict[str, Any]


@dataclass(frozen=True)
class PreparationConfig:
    seed: int
    validation_size: int
    train_size: int | None
    test_size: int | None
    revision: str
    output_dir: Path

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(self.validation_size) is not int or self.validation_size <= 0:
            raise ValueError("validation_size must be a positive integer")
        for name in ("train_size", "test_size"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be null (all) or a positive integer")
        if not isinstance(self.revision, str) or not self.revision.strip():
            raise ValueError("revision must be a nonempty Hub ref or commit SHA")


def load_preparation_config(path: Path) -> PreparationConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    keys = {
        "schema_version",
        "seed",
        "validation_size",
        "train_size",
        "test_size",
        "revision",
        "output_dir",
    }
    if not isinstance(raw, dict) or set(raw) != keys:
        raise ValueError(f"Preparation config requires exactly {sorted(keys)}")
    if type(raw["schema_version"]) is not int or raw.pop("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    if not isinstance(raw["output_dir"], str) or not raw["output_dir"].strip():
        raise ValueError("output_dir must be a nonempty path")
    raw["output_dir"] = (path.resolve().parent / raw["output_dir"]).resolve()
    return PreparationConfig(**raw)


def load_source(revision: str) -> tuple[Mapping[str, Sequence[Record]], str]:
    """Resolve a moving ref once, then load train and test at that exact commit.

    A supplied commit skips ref resolution and can use an already populated HF cache.
    Imports are lazy so unit tests and the existing dry run need no data dependencies.
    """
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ValueError("Install dataset dependencies with pip install -e '.[data]'") from exc
    resolved = revision
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        resolved = HfApi().dataset_info(DATASET, revision=revision).sha
    if not resolved or not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise ValueError("Hub did not return an immutable dataset commit SHA")
    source = load_dataset(DATASET, CONFIGURATION, revision=resolved)
    return source, resolved


def original_problem_id(question: str) -> str:
    """Versioned content ID, independent of row order, subset size and translation.

    Hash exact UTF-8 question bytes; edits to the original text create a new identity.
    Identical questions across official splits have the same ID and are quarantined.
    """
    return "gsm8k:qsha256:" + hashlib.sha256(question.encode("utf-8")).hexdigest()


def assert_split_integrity(splits: Mapping[str, Sequence[Record]]) -> None:
    """Runtime assertions remain active under python -O (never use bare assert)."""
    seen: dict[str, str] = {}
    for split, rows in splits.items():
        for row in rows:
            problem_id = row["original_problem_id"]
            if problem_id in seen:
                raise ValueError(f"ID {problem_id} repeated in {seen[problem_id]} and {split}")
            expected_source = "test" if split == "test" else "train"
            if row["source_split"] != expected_source or row["split"] != split:
                raise ValueError(f"incorrect split provenance for {problem_id}")
            seen[problem_id] = split


def _rank(rows: Sequence[Record], seed: int, purpose: str) -> list[Record]:
    def key(row: Record) -> tuple[str, str]:
        identity = row["original_problem_id"]
        digest = hashlib.sha256(f"{seed}:{purpose}:{identity}".encode()).hexdigest()
        return digest, identity

    return sorted(rows, key=key)


def _write_jsonl(root: Path, relative: str, rows: Sequence[Record]) -> Record:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True)
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
        + "\n"
        for row in rows
    )
    path.write_text(content, encoding="utf-8")
    return {
        "path": relative,
        "count": len(rows),
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }


def prepare_data(config: PreparationConfig) -> Path:
    """Export an audit report even when individual rows or requested sizes are invalid.

    Any rejected row makes status 'error'; the CLI then exits nonzero. No silent
    replacement or shrinking of requested subset sizes is performed.
    """
    if config.output_dir.exists():
        raise FileExistsError(f"Output already exists: {config.output_dir}; choose a new directory")
    source, revision = load_source(config.revision)
    if set(source) != {"train", "test"}:
        raise ValueError("GSM8K main must provide exactly official train and test splits")
    questions = Counter(
        row["question"]
        for rows in source.values()
        for row in rows
        if isinstance(row.get("question"), str) and row["question"].strip()
    )
    parsed: dict[str, list[Record]] = {"train": [], "test": []}
    errors: list[Record] = []
    source_hashes = {}
    for source_split in ("train", "test"):
        digest = hashlib.sha256()
        for index, raw in enumerate(source[source_split]):
            digest.update((json.dumps(raw, sort_keys=True, ensure_ascii=False) + "\n").encode())
            question = raw.get("question")
            identity = original_problem_id(question) if isinstance(question, str) else None
            try:
                if not isinstance(question, str) or not question.strip():
                    raise ValueError("question must be a nonempty string")
                if questions[question] != 1:
                    raise ValueError("duplicate original question; all occurrences quarantined")
                rationale, gold = parse_gsm8k_answer(raw.get("answer"))
                parsed[source_split].append(
                    {
                        "original_problem_id": identity,
                        "source_split": source_split,
                        "source_index": index,
                        "original_question": question,
                        "original_answer": raw["answer"],
                        "rationale": rationale,
                        "gold_answer": gold,
                        "language": "en",
                    }
                )
            except ValueError as exc:
                errors.append(
                    {
                        "source_split": source_split,
                        "source_index": index,
                        "original_problem_id": identity,
                        "raw_record": raw,
                        "error": str(exc),
                    }
                )
        source_hashes[source_split] = digest.hexdigest()
    ranked = _rank(parsed["train"], config.seed, "validation-v1")
    validation = ranked[: config.validation_size]
    training = ranked[config.validation_size :]
    splits = {"train": training, "validation": validation, "test": parsed["test"]}
    for split, rows in splits.items():
        for row in rows:
            row["split"] = split
    assert_split_integrity(splits)
    failures = []
    if errors:
        failures.append(f"{len(errors)} source rows rejected; see errors.jsonl")
    if len(validation) != config.validation_size or not training:
        failures.append("validation_size must leave at least one valid training question")
    subsets = {}
    for split, size in (("train", config.train_size), ("test", config.test_size)):
        rows = _rank(splits[split], config.seed, f"subset-{split}-v1")
        if size is not None and size > len(rows):
            failures.append(f"{split}_size={size} exceeds {len(rows)} valid available questions")
        else:
            subsets[split] = rows if size is None else rows[:size]
    assert_split_integrity({**subsets, "validation": validation})
    for split in ("train", "test"):
        rejected = sum(e["source_split"] == split for e in errors)
        if len(parsed[split]) + rejected != len(source[split]):
            raise ValueError(f"Source accounting failed for {split}")
    config.output_dir.mkdir(parents=True, exist_ok=False)
    split_artifacts = {
        name: _write_jsonl(config.output_dir, f"splits/{name}.jsonl", rows)
        for name, rows in splits.items()
    }
    # On failure, retain canonical valid records and errors, but no runnable subset views.
    subset_artifacts = (
        {}
        if failures
        else {
            name: _write_jsonl(config.output_dir, f"subsets/{name}.jsonl", rows)
            for name, rows in subsets.items()
        }
    )
    error_artifact = _write_jsonl(config.output_dir, "errors.jsonl", errors)
    try:
        datasets_version = version("datasets")
    except PackageNotFoundError:
        datasets_version = None  # Fixture-injected unit tests do not require datasets.
    manifest = {
        "schema_version": 1,
        "preparation_version": 1,
        "package_version": __version__,
        "datasets_version": datasets_version,
        "status": "error" if failures else "ok",
        "failures": failures,
        "dataset": {
            "name": DATASET,
            "configuration": CONFIGURATION,
            "requested_revision": config.revision,
            "resolved_revision": revision,
            "license": "MIT",
            "license_url": "https://github.com/openai/grade-school-math/blob/master/LICENSE",
            "source_counts": {k: len(v) for k, v in source.items()},
            "source_sha256": source_hashes,
        },
        "config": {**asdict(config), "output_dir": str(config.output_dir)},
        "id_algorithm": "gsm8k:qsha256: + SHA256(exact original question UTF-8)",
        "selection_algorithm": "SHA256(seed:purpose:original_problem_id), ascending; v1",
        "numeric_format": "US comma thousands / dot decimal; canonical exact decimal string",
        "splits": split_artifacts,
        "subsets": subset_artifacts,
        "errors": error_artifact,
        "integrity": {"disjoint_original_ids": True, "source_rows_accounted_for": True},
    }
    path = config.output_dir / "manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    configure_logging().info("preparation_failed" if failures else "preparation_completed")
    return path
