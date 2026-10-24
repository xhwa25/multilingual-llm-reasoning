"""Optional plotting, imported only when requested by future analysis code."""

from pathlib import Path


def plot_accuracy(accuracies: dict[str, float], output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    try:
        axis.bar(list(accuracies), list(accuracies.values()))
        axis.set(xlabel="Language", ylabel="Exact-match accuracy", ylim=(0, 1))
        figure.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output)
    finally:
        plt.close(figure)


# Okabe–Ito colors plus hatches/markers: no information relies on color alone.
COLORS = ("#0072B2", "#D55E00", "#009E73")
HATCHES = ("", "///", "xx")
NAMES = {"base": "Base", "english_lora": "English LoRA", "multilingual_lora": "Multilingual LoRA"}
LANGUAGE_NAMES = {"en": "English", "zh": "Chinese", "de": "German", "ja": "Japanese"}


def plot_metrics(
    metrics_path: Path, output_dir: Path, *, training_logs: dict[str, Path] | None = None
) -> dict:
    """Render saved metrics with Agg; do not run inference or invent uncertainty."""
    import json
    import math
    import textwrap
    import warnings

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ValueError("Install plots with pip install -e '.[analysis]'") from exc
    from .comparison_metrics import CONDITIONS
    from .error_analysis import file_hash, read_jsonl
    from .translation import LANGUAGES

    if output_dir.exists():
        raise ValueError("Plot output exists; choose a new directory")
    report = json.loads(metrics_path.read_text())
    summary = report["summary"]
    table = {(r["condition_id"], r["language"]): r for r in summary}
    if len(table) != len(summary) or set(table) != {
        (c, lang) for c in CONDITIONS for lang in LANGUAGES
    }:
        raise ValueError("Plot summary needs exactly three conditions × four languages")

    def finite(value):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Metric must be a finite number")
        return float(value)

    for row in summary:
        if type(row["n_expected"]) is not int or row["n_expected"] <= 0:
            raise ValueError("Positive explicit denominator required")
        for key in ("accuracy", "parse_failure_rate"):
            if not 0 <= finite(row[key]) <= 1:
                raise ValueError("Rates must lie in [0,1]")
        finite(row["delta_vs_base"])
    if len({r["n_expected"] for r in summary}) != 1:
        raise ValueError("Paired comparison denominators must match")
    gaps = {(r["condition_id"], r["other_language"]): r for r in report["english_gaps"]}
    consistency = {r["condition_id"]: r for r in report["consistency"]}
    if len(gaps) != len(report["english_gaps"]) or set(gaps) != {
        (c, lang) for c in CONDITIONS for lang in LANGUAGES if lang != "en"
    }:
        raise ValueError("Missing or duplicate language gaps")
    if len(consistency) != len(report["consistency"]) or set(consistency) != set(CONDITIONS):
        raise ValueError("Missing or duplicate consistency conditions")
    for row in [*gaps.values(), *consistency.values()]:
        if row["n_originals"] != summary[0]["n_expected"]:
            raise ValueError("Question-level denominators must match the paired test set")
    for row in consistency.values():
        if not 0 <= finite(row["estimate"]) <= 1:
            raise ValueError("Consistency must lie in [0,1]")
    for row in gaps.values():
        if not -1 <= finite(row["estimate"]) <= 1:
            raise ValueError("Accuracy gaps must lie in [-1,1]")
    all_rows = summary + list(gaps.values()) + list(consistency.values())
    for row in all_rows:
        for prefix in ("", "delta_", "accuracy_", "parse_failure_"):
            low, high = prefix + "ci95_low", prefix + "ci95_high"
            if (low in row) != (high in row):
                raise ValueError("Interval requires both bounds")
            if low in row and finite(row[low]) > finite(row[high]):
                raise ValueError("Reversed uncertainty interval")
        if "estimate" in row:
            finite(row["estimate"])
    if not training_logs:
        conventional_root = metrics_path.parent.parent / "training"
        training_logs = {
            c: conventional_root / c / "full/training_log.jsonl"
            for c in CONDITIONS[1:]
            if (conventional_root / c / "full/training_log.jsonl").is_file()
        }
    log_data = {}
    for condition, path in (training_logs or {}).items():
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown training-log condition: {condition}")
        records = read_jsonl(path)
        for row in records:
            for key in ("loss", "eval_loss", "initial_eval_loss"):
                if key in row:
                    finite(row[key])
                    if finite(row.get("step")) < 0:
                        raise ValueError("Loss step must be nonnegative")
        log_data[condition] = records
    output_dir.mkdir(parents=True)
    n = summary[0]["n_expected"]
    tag = (
        "MOCK — diagnostics only"
        if report.get("mock")
        else (
            "Saved evaluation"
            if report.get("research_ready")
            else (
                "Provisional translated evaluation"
                if report.get("complete")
                else "Incomplete evaluation"
            )
        )
    )
    artifacts = []
    styles = {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "svg.hashsalt": "multilingual-reasoning-plots-v1",
        "savefig.dpi": 300,
    }

    def interval(ax, x, row, prefix="", scale=100):
        low, high = prefix + "ci95_low", prefix + "ci95_high"
        if low in row:
            lo, hi = row[low] * scale, row[high] * scale
            ax.vlines(x, lo, hi, color="#222222", linewidth=1.3, zorder=4)
            ax.hlines([lo, hi], x - 0.045, x + 0.045, color="#222222", linewidth=1.3, zorder=4)

    def save(fig, ax, slug, title, description):
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#DADADA", linewidth=0.6)
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.22, top=0.73)
        fig.suptitle(title, x=0.12, y=0.97, ha="left", fontsize=15, fontweight="bold")
        fig.text(
            0.12,
            0.905,
            (
                f"{tag} · training logs"
                if slug == "training_validation_loss"
                else f"{tag} · {n} original questions per language"
            ),
            fontsize=10,
        )
        fig.text(
            0.12,
            0.06,
            "\n".join(textwrap.fill(line, 100) for line in description.splitlines()),
            fontsize=9,
            va="bottom",
            wrap=True,
        )
        for extension in ("png", "svg"):
            path = output_dir / f"{slug}.{extension}"
            metadata = (
                {"Title": title, "Description": description, "Date": None}
                if extension == "svg"
                else {"Title": title, "Description": description}
            )
            fig.savefig(path, metadata=metadata)
            artifacts.append(
                {"path": path.name, "sha256": file_hash(path), "description": description}
            )
        plt.close(fig)

    # Save warnings as QA evidence; font/layout issues should not go unnoticed.
    with warnings.catch_warnings(record=True) as caught, plt.rc_context(styles):
        warnings.simplefilter("always")
        for key, slug, title, ylabel, interval_prefix in (
            (
                "accuracy",
                "accuracy_by_language",
                "Accuracy by language and model condition",
                "Exact-match accuracy (%)",
                "accuracy_",
            ),
            (
                "delta_vs_base",
                "improvement_by_language",
                "Accuracy change relative to the base model",
                "Difference from base (percentage points)",
                "delta_",
            ),
            (
                "parse_failure_rate",
                "parse_failure_rates",
                "Parse-failure rates by language",
                "Parse failures / expected questions (%)",
                "parse_failure_",
            ),
        ):
            fig, ax = plt.subplots(figsize=(9, 5.4))
            conditions = CONDITIONS[1:] if key == "delta_vs_base" else CONDITIONS
            width = 0.22
            for j, c in enumerate(conditions):
                ci = CONDITIONS.index(c)
                xs = [i + (j - (len(conditions) - 1) / 2) * width for i in range(4)]
                values = [table[c, lang][key] * 100 for lang in LANGUAGES]
                bars = ax.bar(
                    xs,
                    values,
                    width,
                    color=COLORS[ci],
                    hatch=HATCHES[ci],
                    edgecolor="#222222",
                    linewidth=0.6,
                    label=NAMES[c],
                )
                if key == "delta_vs_base":
                    ax.scatter(
                        xs,
                        values,
                        marker=("o", "s", "D")[ci],
                        color=COLORS[ci],
                        edgecolor="#222222",
                        s=32,
                        zorder=5,
                    )
                for x, lang in zip(xs, LANGUAGES, strict=True):
                    interval(ax, x, table[c, lang], interval_prefix)
                if not any(interval_prefix + "ci95_low" in table[c, lang] for lang in LANGUAGES):
                    ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=3)
            ax.set_xticks(range(4), [LANGUAGE_NAMES[lang] for lang in LANGUAGES])
            ax.set_ylabel(ylabel)
            ax.legend(
                loc="lower left",
                bbox_to_anchor=(0, 1.02),
                ncol=len(conditions),
                frameon=False,
                fontsize=10,
            )
            if key != "delta_vs_base":
                ax.set_ylim(0, 105)
            else:
                ax.axhline(0, color="#333333", linewidth=1)
                ax.margins(y=0.2)
            has_ci = any(
                interval_prefix + "ci95_low" in table[c, lang]
                for c in conditions
                for lang in LANGUAGES
            )
            note = (
                "Whiskers: saved paired bootstrap 95% intervals; exploratory, unadjusted."
                if has_ci
                else "No uncertainty intervals were saved for this metric; none are invented."
            )
            save(
                fig,
                ax,
                slug,
                title,
                note
                + (
                    " All expected questions remain in the denominator; inference failures are "
                    "separate from parse failures."
                ),
            )
        fig, ax = plt.subplots(figsize=(9, 5.4))
        others = [lang for lang in LANGUAGES if lang != "en"]
        for j, c in enumerate(CONDITIONS):
            xs = [i + (j - 1) * 0.22 for i in range(3)]
            ax.bar(
                xs,
                [gaps[c, lang]["estimate"] * 100 for lang in others],
                0.22,
                color=COLORS[j],
                hatch=HATCHES[j],
                edgecolor="#222222",
                linewidth=0.6,
                label=NAMES[c],
            )
            ax.scatter(
                xs,
                [gaps[c, lang]["estimate"] * 100 for lang in others],
                marker=("o", "s", "D")[j],
                color=COLORS[j],
                edgecolor="#222222",
                s=32,
                zorder=5,
            )
            for x, lang in zip(xs, others, strict=True):
                interval(ax, x, gaps[c, lang])
        ax.axhline(0, color="#333333", linewidth=1)
        ax.set_xticks(range(3), [f"English − {LANGUAGE_NAMES[lang]}" for lang in others])
        ax.set_ylabel("Accuracy gap (percentage points)")
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=False, fontsize=10)
        ax.margins(y=0.2)
        save(
            fig,
            ax,
            "english_language_gaps",
            "English versus other-language accuracy gaps",
            (
                "Positive values mean higher English accuracy. Whiskers show saved 95% "
                "intervals where available.\nLanguages are paired by original question; gaps "
                "do not establish their cause."
            ),
        )
        fig, ax = plt.subplots(figsize=(9, 5.4))
        for i, c in enumerate(CONDITIONS):
            row = consistency[c]
            ax.bar(
                i,
                row["estimate"] * 100,
                0.55,
                color=COLORS[i],
                hatch=HATCHES[i],
                edgecolor="#222222",
                linewidth=0.6,
            )
            interval(ax, i, row)
        ax.set_xticks(range(3), [NAMES[c] for c in CONDITIONS])
        ax.set_ylim(0, 105)
        ax.set_ylabel("Questions correct in all four languages (%)")
        save(
            fig,
            ax,
            "cross_language_consistency",
            "Cross-language answer consistency",
            (
                "Denominator: original questions, not language variants. Whiskers: saved 95% "
                "intervals where available.\nA question counts only when all four answers "
                "are correct, including successful parsing."
            ),
        )
        if log_data:
            fig, ax = plt.subplots(figsize=(9, 5.4))
            lines = 0
            for condition, rows in sorted(log_data.items()):
                ci = CONDITIONS.index(condition)
                # Preserve step resets across resume sessions; no smoothing.
                segments, current, previous = [], [], -1
                for row in rows:
                    if not any(k in row for k in ("loss", "eval_loss")):
                        continue
                    if row["step"] < previous:
                        segments.append(current)
                        current = []
                    current.append(row)
                    previous = row["step"]
                segments.append(current)
                for key, style, marker, label in (
                    ("loss", "-", "o", "train"),
                    ("eval_loss", "--", "s", "validation"),
                ):
                    shown = False
                    for segment in segments:
                        values = [r for r in segment if key in r]
                        if values:
                            ax.plot(
                                [r["step"] for r in values],
                                [r[key] for r in values],
                                color=COLORS[ci],
                                linestyle=style,
                                marker=marker,
                                markersize=3,
                                label=f"{NAMES[condition]} · {label}" if not shown else None,
                            )
                            shown = True
                            lines += 1
            if lines:
                ax.set(xlabel="Optimizer step", ylabel="Loss (logged value)")
                ax.legend(
                    loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False, fontsize=9
                )
                save(
                    fig,
                    ax,
                    "training_validation_loss",
                    "Training and validation loss",
                    (
                        "Observed logs only; no smoothing or uncertainty estimated. "
                        "Step resets are "
                        "separate segments.\nTraining loss may be interval-averaged; "
                        "validation loss "
                        "uses the held-out validation set."
                    ),
                )
            else:
                plt.close(fig)
        warning_text = sorted({str(w.message) for w in caught})
    manifest = {
        "schema_version": 1,
        "source": {"path": str(metrics_path.resolve()), "sha256": file_hash(metrics_path)},
        "training_logs": {
            c: {"path": str(p.resolve()), "sha256": file_hash(p)}
            for c, p in (training_logs or {}).items()
        },
        "matplotlib_version": matplotlib.__version__,
        "mock": report.get("mock", False),
        "n_originals": n,
        "artifacts": artifacts,
        "warnings": warning_text,
        "loss_plot_skip_reason": None
        if any(a["path"] == "training_validation_loss.png" for a in artifacts)
        else "No train/validation loss samples supplied or discovered",
        "loss_plot_created": any(a["path"] == "training_validation_loss.png" for a in artifacts),
        "uncertainty_policy": (
            "Only saved intervals are plotted; absent intervals are not estimated."
        ),
        "accessibility": (
            "Colorblind-friendly palette, hatches, markers, explicit labels; SVG text "
            "remains text. Descriptions accompany every figure."
        ),
    }
    (output_dir / "plot_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "source_metrics.json").write_bytes(metrics_path.read_bytes())
    (output_dir / "figures.md").write_text(
        "# Figures\n\n"
        + tag
        + "\n\n"
        + "\n\n".join(
            f"![{a['description'].replace(chr(10), ' ')}]({a['path']})"
            for a in artifacts
            if a["path"].endswith(".png")
        )
        + "\n"
    )
    return manifest


def plot_baseline_metrics(metrics_path: Path, output_dir: Path) -> dict:
    """Render accessible PNG/SVG summaries for one four-language baseline."""
    import json
    import math
    import warnings

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ValueError("Install plots with pip install -e '.[analysis]'") from exc
    from .error_analysis import file_hash
    from .translation import LANGUAGES

    if output_dir.exists():
        raise ValueError("Plot output exists; choose a new directory")
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    if report.get("analysis_type") != "single-condition-cross-lingual-baseline":
        raise ValueError("Expected single-condition baseline metrics")
    n = report.get("n_originals")
    language_metrics = report.get("language_metrics", {})
    if type(n) is not int or n < 1 or set(language_metrics) != set(LANGUAGES):
        raise ValueError("Metrics must contain all four languages and a positive denominator")
    for row in language_metrics.values():
        if row.get("n_expected") != n:
            raise ValueError("Language denominators differ")
        for key in ("accuracy", "parse_failure_rate"):
            value = row.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid rate for {key}")
    gaps = report.get("english_minus_other_language_accuracy_gaps", {})
    if set(gaps) != set(LANGUAGES) - {"en"}:
        raise ValueError("Missing English-language gaps")
    consistency = report.get("cross_language_consistency", {})
    if any(
        type(consistency.get(key)) not in (int, float) or not 0 <= consistency[key] <= 1
        for key in ("all_four_correct", "same_parsed_answer")
    ):
        raise ValueError("Invalid consistency metrics")

    names = {"en": "English", "zh": "Chinese", "de": "German", "ja": "Japanese"}
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    hatches = ("", "///", "xx", "\\\\")
    artifacts = []
    descriptions = []
    warning_rows = []
    styles = {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "svg.hashsalt": "multilingual-reasoning-baseline-v1",
        "savefig.dpi": 300,
    }

    def save(slug, title, description, labels, values, ylabel, limits=None):
        fig, ax = plt.subplots(figsize=(9, 5.4))
        bars = ax.bar(range(len(labels)), values, color=colors[: len(labels)], edgecolor="#222222")
        for bar, hatch, value in zip(bars, hatches[: len(labels)], values, strict=True):
            bar.set_hatch(hatch)
            label = f"{value * 100:.0f}%" if limits == (0, 1) else f"{value * 100:+.0f} pp"
            offset = 0.015 if limits == (0, 1) else 0.03
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                label,
                ha="center",
                va="bottom",
                fontsize=10,
            )
        ax.set_xticks(range(len(labels)), labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold", pad=27)
        ax.text(
            0, 1.02, f"PRELIMINARY · {n} original questions", transform=ax.transAxes, fontsize=9
        )
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#DADADA", linewidth=0.6)
        if limits is not None:
            ax.set_ylim(*limits)
        else:
            bound = max(0.1, *(abs(value) for value in values))
            ax.set_ylim(-bound * 1.35, bound * 1.35)
            ax.axhline(0, color="#222222", linewidth=0.8)
        fig.subplots_adjust(left=0.13, right=0.97, top=0.84, bottom=0.17)
        with warnings.catch_warnings(record=True) as caught, plt.rc_context(styles):
            warnings.simplefilter("always")
            for extension in ("png", "svg"):
                path = output_dir / f"{slug}.{extension}"
                metadata = {"Title": title, "Description": description}
                if extension == "svg":
                    metadata["Date"] = None
                fig.savefig(path, metadata=metadata)
                artifacts.append({"path": path.name, "sha256": file_hash(path)})
            warning_rows.extend(
                {"category": item.category.__name__, "message": str(item.message)}
                for item in caught
            )
        plt.close(fig)
        descriptions.append(f"## {title}\n\n{description}\n")

    output_dir.mkdir(parents=True)
    labels = [names[language] for language in LANGUAGES]
    save(
        "baseline_accuracy_by_language",
        "Base-model accuracy by language",
        f"Exact-match accuracy. Denominator: {n} questions per language. "
        "No uncertainty intervals were computed.",
        labels,
        [language_metrics[language]["accuracy"] for language in LANGUAGES],
        "Accuracy",
        (0, 1),
    )
    save(
        "baseline_parse_failure_rates",
        "Answer parse-failure rate by language",
        f"Parse failures divided by all {n} expected questions per language.",
        labels,
        [language_metrics[language]["parse_failure_rate"] for language in LANGUAGES],
        "Parse-failure rate",
        (0, 1),
    )
    others = [language for language in LANGUAGES if language != "en"]
    save(
        "baseline_english_language_gaps",
        "English minus other-language accuracy",
        "Paired language accuracy differences on the same original questions. "
        "Positive values mean higher English accuracy.",
        [f"English − {names[language]}" for language in others],
        [gaps[language] for language in others],
        "Accuracy difference (percentage points)",
    )
    save(
        "baseline_cross_language_consistency",
        "Question-level cross-language consistency",
        f"All-four-correct and same-parsed-answer rates, each over {n} original questions.",
        ["All four correct", "Same parsed answer"],
        [consistency["all_four_correct"], consistency["same_parsed_answer"]],
        "Original-question rate",
        (0, 1),
    )
    manifest = {
        "schema_version": 1,
        "source_metrics": {"path": str(metrics_path.resolve()), "sha256": file_hash(metrics_path)},
        "n_originals": n,
        "preliminary": True,
        "artifacts": artifacts,
        "warnings": warning_rows,
        "accessibility": (
            "Colorblind-friendly colors, distinct hatches, direct labels, figure descriptions, "
            "SVG text remains text."
        ),
    }
    (output_dir / "plot_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "figures.md").write_text(
        "# Preliminary baseline figures\n\n" + "\n".join(descriptions), encoding="utf-8"
    )
    return manifest
