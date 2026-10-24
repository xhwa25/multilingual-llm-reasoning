"""Matched English and multilingual-question SFT artifacts with English targets."""

import hashlib
import json
import math
import statistics
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .logging import configure_logging
from .numeric import parse_gsm8k_answer
from .preparation import assert_split_integrity, original_problem_id
from .scoring import parse_final_answer
from .sft_config import SFTConfig
from .sft_tokenization import ChatTokenizer, load_sft_tokenizer, tokenize_sft
from .translation import LANGUAGES

Record = dict[str, Any]
SFT_PROMPT_VERSION = "sft-english-worked-solution-v1"
SFT_INSTRUCTION = (
    "Solve the problem. Give the worked solution in English, preserving the equations. "
    "End with a separate line in the format FINAL_ANSWER: <number>, using a single "
    "signed decimal number with a dot decimal point and no units."
)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _artifact(manifest_path: Path, artifact: Record) -> list[Record]:
    root = manifest_path.parent.resolve()
    path = (root / artifact["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Artifact path escapes its manifest directory")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != artifact["sha256"]:
        raise ValueError(f"Artifact hash mismatch: {path.name}")
    rows = [json.loads(line) for line in data.splitlines()]
    if len(rows) != artifact["count"]:
        raise ValueError("Artifact record count mismatch")
    return rows


def _rank(rows: list[Record], seed: int, purpose: str) -> list[Record]:
    return sorted(rows, key=lambda row: _hash(f"{seed}:{purpose}:{row['original_problem_id']}"))


def load_sft_sources(
    config: SFTConfig, *, english_only: bool = False
) -> tuple[list[Record], list[Record], dict, Record]:
    prep_bytes = config.preparation_manifest.read_bytes()
    prep = json.loads(prep_bytes)
    if prep.get("schema_version") != 1 or prep.get("status") != "ok":
        raise ValueError("SFT requires a successful preparation manifest")
    canonical = {
        name: _artifact(config.preparation_manifest, prep["splits"][name])
        for name in ("train", "validation", "test")
    }
    assert_split_integrity(canonical)
    lookup = {row["original_problem_id"]: row for rows in canonical.values() for row in rows}
    for identity, row in lookup.items():
        if identity != original_problem_id(row["original_question"]):
            raise ValueError("Original question ID mismatch")
        rationale, gold = parse_gsm8k_answer(row["original_answer"])
        if rationale != row["rationale"] or gold != row["gold_answer"]:
            raise ValueError("Prepared rationale or gold differs from original GSM8K solution")
    pool = _artifact(config.preparation_manifest, prep["subsets"]["train"])
    assert_split_integrity({"train": pool})
    for row in pool:
        if row != lookup.get(row["original_problem_id"]):
            raise ValueError("Training subset differs from canonical prepared row")
    if len(pool) < config.train_size or len(canonical["validation"]) < config.validation_size:
        raise ValueError(
            "Requested SFT size exceeds available prepared data; no replacement sampling"
        )
    training = _rank(pool, config.seed, "sft-train-v1")[: config.train_size]
    validation = _rank(canonical["validation"], config.seed, "sft-validation-v1")[
        : config.validation_size
    ]
    prep_hash = hashlib.sha256(prep_bytes).hexdigest()
    provenance = {
        "preparation_manifest_sha256": prep_hash,
        "translation_manifest_sha256": None,
        "mock_questions": False,
        "all_validation_ids": [row["original_problem_id"] for row in canonical["validation"]],
        "all_test_ids": [row["original_problem_id"] for row in canonical["test"]],
        "allowed_train_ids": [row["original_problem_id"] for row in canonical["train"]],
        "dataset": prep["dataset"],
    }
    if english_only:
        return training, validation, {}, provenance
    translated_bytes = config.translation_manifest.read_bytes()
    translated = json.loads(translated_bytes)
    if translated.get("schema_version") != 1 or not translated.get("complete"):
        raise ValueError("SFT requires a complete question-translation manifest")
    prep_hash = hashlib.sha256(prep_bytes).hexdigest()
    if translated.get("input_manifest_sha256") != prep_hash:
        raise ValueError("Question translations came from a different preparation manifest")
    variants = _artifact(config.translation_manifest, translated["artifacts"]["variants"])
    by_pair = {}
    for row in variants:
        pair = (row["original_problem_id"], row["language"])
        original = lookup.get(pair[0])
        if pair in by_pair or row["language"] not in LANGUAGES:
            raise ValueError("Duplicate/unsupported translated variant")
        if original is None or any(
            row.get(key) != original[key] for key in ("split", "source_split", "gold_answer")
        ):
            raise ValueError("Translation identity, split, or gold does not match original")
        if row.get("source_text") != original["original_question"]:
            raise ValueError("Translation source text differs from original")
        if not isinstance(row.get("translated_text"), str) or not row["translated_text"].strip():
            raise ValueError("Translation has missing question text")
        if pair[1] == "en" and row["translated_text"] != original["original_question"]:
            raise ValueError("English question was changed")
        by_pair[pair] = row
    for index, row in enumerate(training):
        lang = LANGUAGES[index % 4]
        if lang == "en":
            continue
        variant = by_pair.get((row["original_problem_id"], lang))
        if variant is None:
            raise ValueError(
                f"Missing assigned training variant: {row['original_problem_id']} / {lang}"
            )
        ready = not translated.get("mock", False) and variant.get("qa_status") == "human_approved"
        if not ready and not config.allow_unreviewed_questions:
            raise ValueError(
                "Assigned training translations are mock/unreviewed; production SFT refused"
            )
    provenance = {
        "preparation_manifest_sha256": prep_hash,
        "translation_manifest_sha256": hashlib.sha256(translated_bytes).hexdigest(),
        "mock_questions": bool(translated.get("mock", False)),
        "all_validation_ids": [row["original_problem_id"] for row in canonical["validation"]],
        "all_test_ids": [row["original_problem_id"] for row in canonical["test"]],
        "allowed_train_ids": [row["original_problem_id"] for row in canonical["train"]],
        "dataset": prep["dataset"],
    }
    return training, validation, by_pair, provenance


def _example(
    original: Record,
    language: str,
    question: str,
    condition: str,
    question_provenance: Record,
    tokenizer: ChatTokenizer,
    max_length: int,
) -> Record:
    # Original English worked solution, including every equation/calculator annotation.
    target = original["rationale"] + "\nFINAL_ANSWER: " + original["gold_answer"]
    if parse_final_answer(target).answer != original["gold_answer"]:
        raise ValueError("Target final answer does not match original gold")
    messages = [
        {"role": "system", "content": SFT_INSTRUCTION},
        {"role": "user", "content": question},
        {"role": "assistant", "content": target},
    ]
    tokens = tokenize_sft(tokenizer, messages)
    return {
        "original_problem_id": original["original_problem_id"],
        "condition": condition,
        "split": original["split"],
        "source_split": original["source_split"],
        "source_index": original["source_index"],
        "question_language": language,
        "solution_language": "en",
        "solution_policy": "original_english",
        "entire_example_multilingual": False,
        "original_question": original["original_question"],
        "original_answer": original["original_answer"],
        "original_rationale": original["rationale"],
        "original_solution_sha256": _hash(original["original_answer"]),
        "question_text": question,
        "target_text": target,
        "gold_answer": original["gold_answer"],
        "question_provenance": question_provenance,
        "prompt_version": SFT_PROMPT_VERSION,
        "messages": messages,
        **tokens,
        "length_status": "ok" if tokens["token_length"] <= max_length else "overlength",
    }


def validate_sft_pairs(
    english: list[Record],
    multilingual: list[Record],
    validation: list[Record],
    allowed_train_ids: set[str],
    reserved_ids: set[str],
    *,
    english_only: bool = False,
    expected_language_counts: dict[str, int] | None = None,
) -> None:
    sets = []
    for group in (english, multilingual, validation):
        identities = [row["original_problem_id"] for row in group]
        if len(set(identities)) != len(identities):
            raise ValueError("Duplicate original ID in SFT condition")
        sets.append(set(identities))
        for row in group:
            rationale, gold = parse_gsm8k_answer(row["original_answer"])
            expected = rationale + "\nFINAL_ANSWER: " + gold
            if row["solution_language"] != "en" or row["solution_policy"] != "original_english":
                raise ValueError("English original target must be explicitly labeled English")
            if row["target_text"] != expected or row["messages"][-1]["content"] != expected:
                raise ValueError("Original equations/rationale or final answer were modified")
            if parse_final_answer(row["target_text"]).answer != row["gold_answer"]:
                raise ValueError("Final answer mismatch")
            if len(row["input_ids"]) != len(row["labels"]) or len(row["labels"]) != len(
                row["attention_mask"]
            ):
                raise ValueError("Token/label/attention lengths differ")
            n = row["prompt_tokens"]
            if row["labels"][:n] != [-100] * n or row["labels"][n:] != row["input_ids"][n:]:
                raise ValueError("Assistant-only loss mask invalid")
    if (not english_only and sets[0] != sets[1]) or not sets[0] <= allowed_train_ids:
        raise ValueError("Training conditions must match exactly and use only allowed training IDs")
    if sets[0] & reserved_ids or sets[0] & sets[2]:
        raise ValueError("Training/validation/test split leakage")
    if not sets[2] <= reserved_ids:
        raise ValueError("Validation must use reserved holdout IDs")
    if any(row["question_language"] != "en" for row in english + validation):
        raise ValueError("English condition/shared validation must use English questions")
    if any(
        row["split"] != "train" or row["source_split"] != "train" for row in english + multilingual
    ):
        raise ValueError("SFT training must come from official train, not validation or test")
    if any(row["split"] != "validation" or row["source_split"] != "train" for row in validation):
        raise ValueError("SFT validation must come from the prepared training holdout")
    if any(row["question_text"] != row["original_question"] for row in english + validation):
        raise ValueError("English question must be the unmodified original")
    if english_only:
        if multilingual:
            raise ValueError("English-only audit cannot include multilingual rows")
        return
    if expected_language_counts is None:
        if len(english) % 4:
            raise ValueError("Multilingual question languages are not exactly balanced")
        expected_language_counts = {language: len(english) // 4 for language in LANGUAGES}
    if dict(Counter(r["question_language"] for r in multilingual)) != expected_language_counts:
        raise ValueError("Multilingual question language counts differ from approved manifest")
    english_targets = {r["original_problem_id"]: r["target_text"] for r in english}
    if any(
        row["target_text"] != english_targets[row["original_problem_id"]] for row in multilingual
    ):
        raise ValueError("Targets must be identical by ID across conditions")


def _stats(rows: list[Record]) -> Record:
    def summarize(key: str) -> Record:
        values = sorted(row[key] for row in rows)
        return {
            "min": values[0],
            "max": values[-1],
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": values[math.ceil(0.95 * len(values)) - 1],
            "total": sum(values),
        }

    return {
        "count": len(rows),
        "question_language_counts": dict(Counter(r["question_language"] for r in rows)),
        "solution_language_counts": dict(Counter(r["solution_language"] for r in rows)),
        "overlength": sum(r["length_status"] == "overlength" for r in rows),
        **{key: summarize(key) for key in ("token_length", "prompt_tokens", "target_tokens")},
    }


def prepare_sft(
    config: SFTConfig,
    *,
    tokenizer_loader: Callable[[SFTConfig], tuple[ChatTokenizer, Record]] = load_sft_tokenizer,
    english_only: bool = False,
) -> Record:
    if config.output_dir.exists():
        raise FileExistsError("SFT output directory already exists; choose a new directory")
    training, heldout, variants, provenance = load_sft_sources(config, english_only=english_only)
    tokenizer, tokenizer_metadata = tokenizer_loader(config)
    identity = {"provider": "identity", "qa_status": "source_original", "language": "en"}
    english, multilingual, validation = [], [], []
    for index, original in enumerate(training):
        english.append(
            _example(
                original,
                "en",
                original["original_question"],
                "A_english",
                identity,
                tokenizer,
                config.max_seq_length,
            )
        )
        if english_only:
            continue
        language = LANGUAGES[index % 4]
        if language == "en":
            question, qa = original["original_question"], identity
        else:
            translated = variants[(original["original_problem_id"], language)]
            question = translated["translated_text"]
            qa = {
                key: translated.get(key)
                for key in (
                    "translation_provider",
                    "translation_model",
                    "translation_model_version",
                    "translation_provider_version",
                    "timestamp",
                    "prompt_version",
                    "qa_status",
                    "qa_flags",
                    "raw_translation_sha256",
                    "review",
                )
            }
            qa["effective_question_sha256"] = _hash(question)
        multilingual.append(
            _example(
                original,
                language,
                question,
                "B_multilingual_questions_en_targets",
                qa,
                tokenizer,
                config.max_seq_length,
            )
        )
    for row in heldout:
        validation.append(
            _example(
                row,
                "en",
                row["original_question"],
                "shared_validation_english",
                identity,
                tokenizer,
                config.max_seq_length,
            )
        )
    validate_sft_pairs(
        english,
        multilingual,
        validation,
        set(provenance["allowed_train_ids"]),
        set(provenance["all_validation_ids"]) | set(provenance["all_test_ids"]),
        english_only=english_only,
    )
    groups = {
        "english_train": english,
        "multilingual_train": multilingual,
        "validation": validation,
    }
    if english_only:
        groups.pop("multilingual_train")
    diagnostics = {name: _stats(rows) for name, rows in groups.items()}
    diagnostics["multilingual_by_question_language"] = {
        lang: _stats([row for row in multilingual if row["question_language"] == lang])
        for lang in LANGUAGES
        if not english_only
    }
    issues = [
        {
            "original_problem_id": row["original_problem_id"],
            "condition": name,
            "question_language": row["question_language"],
            "error": "overlength",
            "token_length": row["token_length"],
            "max_seq_length": config.max_seq_length,
        }
        for name, rows in groups.items()
        for row in rows
        if row["length_status"] != "ok"
    ]
    unreviewed = [
        row["original_problem_id"]
        for row in multilingual
        if row["question_language"] != "en"
        and (
            provenance["mock_questions"]
            or row["question_provenance"]["qa_status"] != "human_approved"
        )
    ]
    ready = not issues and not unreviewed and not english_only
    report = {
        "schema_version": 1,
        "status": "partial"
        if english_only
        else ("ready" if ready else "preview_not_training_ready"),
        "conditions": {
            "english_train": {
                "status": "validated" if not issues else "invalid",
                "training_ready": not issues,
            },
            "multilingual_train": {
                "status": "missing"
                if english_only
                else ("validated" if ready else "unreviewed_or_invalid"),
                "training_ready": ready,
                "reason": "Reviewed translations required" if english_only else None,
            },
        },
        "training_ready": ready,
        "config": asdict(config),
        "tokenizer": tokenizer_metadata,
        "provenance": provenance,
        "prompt_version": SFT_PROMPT_VERSION,
        "instruction_language": "en",
        "solution_policy": "original_english",
        "condition_B_scope": "Multilingual questions with original English worked-solution targets",
        "solution_translation_performed": False,
        "selection_algorithm": (
            "SHA256(seed:sft-train-v1:original_id) ascending; languages cycle en/zh/de/ja"
        ),
        "selected_train_ids": [row["original_problem_id"] for row in training],
        "selected_validation_ids": [row["original_problem_id"] for row in heldout],
        "language_assignments": {
            r["original_problem_id"]: LANGUAGES[i % 4] for i, r in enumerate(training)
        },
        "unreviewed_question_ids": unreviewed,
        "diagnostics": diagnostics,
        "audit": {
            "matched_original_ids": None if english_only else True,
            "matched_example_counts": None if english_only else True,
            "targets_identical_by_id": None if english_only else True,
            "balanced_question_languages": None if english_only else True,
            "train_validation_test_disjoint": True,
            "original_rationales_preserved": True,
            "truncation_performed": False,
            "examples_dropped": 0,
        },
    }
    config.output_dir.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    for name, rows in {**groups, "issues": issues}.items():
        content = "".join(_json(row) + "\n" for row in rows)
        (config.output_dir / f"{name}.jsonl").write_text(content, encoding="utf-8")
        artifacts[name] = {"path": f"{name}.jsonl", "count": len(rows), "sha256": _hash(content)}
    diagnostic_text = _json(diagnostics) + "\n"
    (config.output_dir / "token_lengths.json").write_text(diagnostic_text, encoding="utf-8")
    artifacts["token_lengths"] = {"path": "token_lengths.json", "sha256": _hash(diagnostic_text)}
    report["artifacts"] = artifacts
    (config.output_dir / "manifest.json").write_text(_json(report) + "\n", encoding="utf-8")
    configure_logging().info(
        "sft_preparation_completed",
        extra={
            "details": {
                "training_examples_per_condition": len(training),
                "validation_examples": len(validation),
                "training_ready": ready,
                "overlength_examples": len(issues),
            }
        },
    )
    return report
