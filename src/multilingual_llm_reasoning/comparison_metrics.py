"""Strict rectangular joins and paired original-question cluster bootstrap."""

import csv
import json
import math
import random
from pathlib import Path

from .baseline_scoring import score_record
from .numeric import normalize_numeric
from .translation import LANGUAGES

CONDITIONS = ("base", "english_lora", "multilingual_lora")


def exact_mcnemar(left: list[int], right: list[int]) -> dict:
    """Exact two-sided paired McNemar test for one variant per independent original."""
    if len(left) != len(right) or not left or any(v not in (0, 1) for v in left + right):
        raise ValueError("McNemar inputs must be equal-length binary paired outcomes")
    left_only = sum(a == 1 and b == 0 for a, b in zip(left, right, strict=True))
    right_only = sum(a == 0 and b == 1 for a, b in zip(left, right, strict=True))
    discordant = left_only + right_only
    p_value = (
        min(
            1.0,
            2
            * sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1))
            / 2**discordant,
        )
        if discordant
        else 1.0
    )
    return {
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "discordant_pairs": discordant,
        "p_value_two_sided": p_value,
    }


def paired_summary(
    rows: list[dict], expected: list[dict], *, seed: int = 42, bootstrap_samples: int = 5000
) -> dict:
    if bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap resamples")
    expected_map = {(r["original_problem_id"], r["language"]): r for r in expected}
    ids = sorted({key[0] for key in expected_map})
    if (
        not ids
        or len(expected_map) != len(expected)
        or set(expected_map) != {(i, lang) for i in ids for lang in LANGUAGES}
    ):
        raise ValueError("Expected set must be unique and complete across four languages")
    for identity in ids:
        if (
            len(
                {
                    normalize_numeric(expected_map[identity, lang]["gold_answer"])
                    for lang in LANGUAGES
                }
            )
            != 1
        ):
            raise ValueError("Expected gold differs across languages")
    joined = {}
    for raw in rows:
        key = (raw.get("condition_id"), raw.get("original_problem_id"), raw.get("language"))
        if key in joined:
            raise ValueError("Duplicate condition/problem/language row")
        if key[0] not in CONDITIONS or key[1:] not in expected_map:
            raise ValueError("Unexpected condition or test variant")
        gold = normalize_numeric(expected_map[key[1:]]["gold_answer"])
        if normalize_numeric(raw.get("gold_answer")) != gold:
            raise ValueError("Gold mismatch in paired join")
        row = score_record(raw)  # Never trust a saved correctness flag.
        if row["parse_status"] == "invalid_gold":
            raise ValueError("Invalid gold")
        joined[key] = row
    if set(joined) != {(c, i, lang) for c in CONDITIONS for i in ids for lang in LANGUAGES}:
        raise ValueError("Missing condition/problem/language rows; no inner-join shrinking allowed")
    for identity, language in expected_map:
        completed = [
            joined[c, identity, language]
            for c in CONDITIONS
            if joined[c, identity, language].get("inference_status", "completed") == "completed"
        ]
        for field in (
            "prompt_messages",
            "decoding",
            "prompt_template_version",
            "chat_template_sha256",
            "model_checkpoint",
            "model_revision",
        ):
            if len({json.dumps(row.get(field), sort_keys=True) for row in completed}) > 1:
                raise ValueError(f"Unpaired inference settings: {field}")
    # One common resample plan preserves pairing across conditions AND languages.
    rng = random.Random(seed)
    n = len(ids)
    samples = [[rng.randrange(n) for _ in range(n)] for _ in range(bootstrap_samples)]

    def interval(values: list[float]) -> dict:
        estimates = sorted(sum(values[i] for i in sample) / n for sample in samples)

        def quantile(p: float) -> float:
            position = (len(estimates) - 1) * p
            lo, hi = math.floor(position), math.ceil(position)
            return estimates[lo] + (position - lo) * (estimates[hi] - estimates[lo])

        return {
            "estimate": sum(values) / n,
            "ci95_low": quantile(0.025),
            "ci95_high": quantile(0.975),
            "n_originals": n,
        }

    outcomes = {
        c: {lang: [int(joined[c, i, lang]["correct"]) for i in ids] for lang in LANGUAGES}
        for c in CONDITIONS
    }
    vectors = {
        c: {
            **outcomes[c],
            "macro_accuracy": [
                sum(outcomes[c][lang][i] for lang in LANGUAGES) / 4 for i in range(n)
            ],
            "all_languages_correct": [
                int(all(outcomes[c][lang][i] for lang in LANGUAGES)) for i in range(n)
            ],
        }
        for c in CONDITIONS
    }
    contrasts = []
    mcnemar = []
    for treatment, reference in (
        ("english_lora", "base"),
        ("multilingual_lora", "base"),
        ("multilingual_lora", "english_lora"),
    ):
        for metric in (*LANGUAGES, "macro_accuracy", "all_languages_correct"):
            contrasts.append(
                {
                    "condition_id": treatment,
                    "reference": reference,
                    "metric": metric,
                    **interval(
                        [
                            a - b
                            for a, b in zip(
                                vectors[treatment][metric], vectors[reference][metric], strict=True
                            )
                        ]
                    ),
                }
            )
        for language in LANGUAGES:
            mcnemar.append(
                {
                    "condition_id": treatment,
                    "reference": reference,
                    "language": language,
                    "n_originals": n,
                    **exact_mcnemar(outcomes[treatment][language], outcomes[reference][language]),
                }
            )
    summary = []
    for c in CONDITIONS:
        for lang in LANGUAGES:
            group = [joined[c, i, lang] for i in ids]
            completed = sum(r.get("inference_status", "completed") == "completed" for r in group)
            failures = sum(
                r["parse_status"] not in ("ok", "inference_error", "pending") for r in group
            )
            delta = next(
                (
                    d
                    for d in contrasts
                    if d["condition_id"] == c and d["reference"] == "base" and d["metric"] == lang
                ),
                None,
            )
            summary.append(
                {
                    "condition_id": c,
                    "language": lang,
                    "n_expected": n,
                    "n_completed": completed,
                    "n_correct": sum(outcomes[c][lang]),
                    "accuracy": sum(outcomes[c][lang]) / n,
                    "n_parse_failures": failures,
                    "parse_failure_rate": failures / n,
                    "parse_failure_rate_among_completed": failures / completed
                    if completed
                    else None,
                    "n_inference_failures_or_pending": n - completed,
                    "delta_vs_base": delta["estimate"] if delta else 0.0,
                    "delta_ci95_low": delta["ci95_low"] if delta else 0.0,
                    "delta_ci95_high": delta["ci95_high"] if delta else 0.0,
                }
            )
    gaps = [
        {
            "condition_id": c,
            "other_language": lang,
            **interval([a - b for a, b in zip(outcomes[c]["en"], outcomes[c][lang], strict=True)]),
        }
        for c in CONDITIONS
        for lang in LANGUAGES
        if lang != "en"
    ]
    consistency = [
        {
            "condition_id": c,
            "n_all_languages_correct": sum(vectors[c]["all_languages_correct"]),
            **interval(vectors[c]["all_languages_correct"]),
        }
        for c in CONDITIONS
    ]
    return {
        "schema_version": 1,
        "summary": summary,
        "contrasts": contrasts,
        "mcnemar": mcnemar,
        "english_gaps": gaps,
        "consistency": consistency,
        "n_originals": n,
        "expected_predictions": n * 12,
        "bootstrap": {
            "seed": seed,
            "resamples": bootstrap_samples,
            "unit": "original_problem_id",
            "method": "paired percentile, linear quantiles, shared resamples",
        },
        "denominator_policy": (
            "Accuracy and primary parse-failure rate divide by all expected originals "
            "per language. Inference failures/pending count incorrect and are reported "
            "separately. No rows dropped."
        ),
        "uncertainty": (
            "Exploratory unadjusted 95% intervals; no multiple-comparison correction. "
            "With 100 originals, each language accuracy changes by 1 percentage point "
            "per question. Bootstrap intervals can be degenerate and omit "
            "translation/model/seed uncertainty; not proof of improvement."
        ),
        "mcnemar_scope": (
            "Exact two-sided McNemar tests are per language, paired by canonical ID. "
            "No pooled 392-row independence assumption or multiple-testing correction."
        ),
    }


def export_summary(report: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("summary", "contrasts", "mcnemar", "english_gaps", "consistency"):
        with (directory / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report[name][0]))
            writer.writeheader()
            writer.writerows(report[name])
    (directory / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def summarize_comparison_file(
    predictions: Path,
    test_set: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    bootstrap_samples: int = 5000,
) -> dict:
    if output_dir.exists():
        raise ValueError("Summary output directory exists; choose a new directory")
    rows = [json.loads(line) for line in predictions.read_bytes().splitlines()]
    expected = [json.loads(line) for line in test_set.read_bytes().splitlines()]
    report = paired_summary(rows, expected, seed=seed, bootstrap_samples=bootstrap_samples)
    report.update(
        mock=all(r.get("backend") == "mock" for r in rows),
        research_ready=False,
        interpretation="Offline rescoring; review original run provenance before interpretation.",
    )
    export_summary(report, output_dir)
    return report
