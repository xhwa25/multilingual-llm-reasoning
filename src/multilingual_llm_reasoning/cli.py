"""Command-line entry points for config validation and the offline smoke run."""

import argparse
import json
from pathlib import Path

import yaml

from .baseline import evaluate_baseline
from .baseline_analysis import analyze_baseline
from .baseline_config import load_baseline_config
from .baseline_scoring import score_file
from .comparison import evaluate_comparison, load_comparison_config
from .comparison_metrics import summarize_comparison_file
from .config import load_config
from .datasets import load_problems
from .error_analysis import analyze_results
from .pipeline import run
from .plotting import plot_baseline_metrics, plot_metrics
from .preparation import load_preparation_config, prepare_data
from .sft import prepare_sft
from .sft_config import load_sft_config
from .training import train_adapter
from .training_config import load_training_config
from .translation_config import load_translation_config
from .translation_pipeline import estimate_translation, translate_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-lingual reasoning research scaffold")
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser(
        "analyze", help="Offline matched error analysis and human-review CSV"
    )
    analyze.add_argument("--input", type=Path, required=True)
    analyze.add_argument("--test-set", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)
    analyze.add_argument("--seed", type=int, default=42)
    analyze.add_argument("--bootstrap-samples", type=int, default=5000)
    plot = commands.add_parser("plot", help="Render saved metrics as accessible PNG/SVG figures")
    plot.add_argument("--metrics", type=Path, required=True)
    plot.add_argument("--output-dir", type=Path, required=True)
    plot.add_argument("--training-log", action="append", default=[], metavar="CONDITION=PATH")
    analyze_baseline_parser = commands.add_parser(
        "analyze-baseline", help="Summarize one real four-language baseline"
    )
    analyze_baseline_parser.add_argument("--input", type=Path, required=True)
    analyze_baseline_parser.add_argument("--test-set", type=Path, required=True)
    analyze_baseline_parser.add_argument("--manifest", type=Path, required=True)
    analyze_baseline_parser.add_argument("--output-dir", type=Path, required=True)
    plot_baseline_parser = commands.add_parser(
        "plot-baseline", help="Render one baseline summary as accessible PNG/SVG figures"
    )
    plot_baseline_parser.add_argument("--metrics", type=Path, required=True)
    plot_baseline_parser.add_argument("--output-dir", type=Path, required=True)
    validate = commands.add_parser("validate", help="Validate configuration and aligned dataset")
    execute = commands.add_parser("run", help="Run the offline smoke pipeline")
    prepare = commands.add_parser("prepare-data", help="Prepare GSM8K splits and subset views")
    translate = commands.add_parser(
        "translate", help="Translate prepared questions with QA and resume"
    )
    translate.add_argument(
        "--estimate-only", action="store_true", help="Offline cold-cache cost estimate"
    )
    translate.add_argument(
        "--allow-paid", action="store_true", help="Explicitly enable API requests"
    )
    translate.add_argument(
        "--corrections", type=Path, help="Import manual CSV or JSONL review events"
    )
    translate.add_argument(
        "--review-only", action="store_true", help="Apply reviews/export; never call providers"
    )
    baseline = commands.add_parser(
        "evaluate-baseline", help="Run a resumable local reasoning baseline"
    )
    baseline.add_argument(
        "--allow-full-run", action="store_true", help="Permit more than 5 original questions"
    )
    baseline.add_argument(
        "--retry-errors", action="store_true", help="Retry recorded inference errors"
    )
    paired = commands.add_parser(
        "summarize-comparison", help="Offline paired metrics from saved predictions"
    )
    paired.add_argument("--input", type=Path, required=True)
    paired.add_argument("--test-set", type=Path, required=True)
    paired.add_argument("--output-dir", type=Path, required=True)
    paired.add_argument("--seed", type=int, default=42)
    paired.add_argument("--bootstrap-samples", type=int, default=5000)
    score = commands.add_parser("score", help="Recompute exact numerical scores from predictions")
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    sft = commands.add_parser("prepare-sft", help="Build matched, audited supervised datasets")
    sft.add_argument(
        "--english-only",
        action="store_true",
        help="Export real English data; mark multilingual data missing",
    )
    sft.add_argument(
        "--complete",
        action="store_true",
        help="Complete an English-only export after translation review",
    )
    train = commands.add_parser("train", help="Guarded CUDA LoRA training and offline preflight")
    train.add_argument("--mode", choices=("dry-run", "smoke", "full"), default="dry-run")
    train.add_argument(
        "--check-hardware", action="store_true", help="Include CUDA/package checks in dry-run"
    )
    train.add_argument("--confirm-gpu-training", action="store_true")
    train.add_argument("--resume", help="latest or a checkpoint path within this condition/mode")
    comparison = commands.add_parser(
        "evaluate-comparison", help="Evaluate base and two adapters on identical test IDs"
    )
    comparison.add_argument("--allow-full-run", action="store_true")
    comparison.add_argument("--retry-errors", action="store_true")
    for command in (validate, execute, prepare, translate, baseline, sft, train, comparison):
        command.add_argument("--config", type=Path, required=True)
    execute.add_argument("--dry-run", action="store_true", help="Use deterministic mock inference")
    args = parser.parse_args(argv)
    try:
        if args.command == "analyze":
            analyze_results(
                args.input,
                args.test_set,
                args.output_dir,
                seed=args.seed,
                bootstrap_samples=args.bootstrap_samples,
            )
            print(args.output_dir / "report.md")
        elif args.command == "analyze-baseline":
            report = analyze_baseline(args.input, args.test_set, args.manifest, args.output_dir)
            print(args.output_dir / "report.md")
            print(
                json.dumps(
                    {"n_predictions": report["n_predictions"], "preliminary": report["preliminary"]}
                )
            )
        elif args.command == "plot-baseline":
            report = plot_baseline_metrics(args.metrics, args.output_dir)
            print(args.output_dir / "plot_manifest.json")
            print(
                json.dumps(
                    {"figures": len(report["artifacts"]), "preliminary": report["preliminary"]}
                )
            )
        elif args.command == "plot":
            logs = {}
            for item in args.training_log:
                if "=" not in item:
                    raise ValueError("--training-log requires CONDITION=PATH")
                condition, value = item.split("=", 1)
                if condition in logs or not value:
                    raise ValueError("Duplicate condition or empty training-log path")
                logs[condition] = Path(value)
            plot_metrics(args.metrics, args.output_dir, training_logs=logs)
            print(args.output_dir / "figures.md")
        elif args.command == "validate":
            config = load_config(args.config)
            problems = load_problems(config.dataset_path, config.languages)
            print(f"Valid configuration; {len(problems)} aligned problem-language rows")
        elif args.command == "prepare-data":
            manifest = prepare_data(load_preparation_config(args.config))
            print(manifest)
            report = json.loads(manifest.read_text(encoding="utf-8"))
            if report["status"] != "ok":
                parser.exit(2, f"error: preparation failed; inspect {manifest}\n")
        elif args.command == "translate":
            config = load_translation_config(args.config)
            if args.estimate_only:
                print(json.dumps(estimate_translation(config), indent=2))
            else:
                report = translate_dataset(
                    config,
                    allow_paid=args.allow_paid,
                    corrections=args.corrections,
                    review_only=args.review_only,
                )
                print(config.output_dir / "manifest.json")
                if not report["complete"] or report["blocked_reason"]:
                    parser.exit(
                        2, "error: translation incomplete or limit reached; inspect QA report\n"
                    )
        elif args.command == "evaluate-baseline":
            config = load_baseline_config(args.config)
            report = evaluate_baseline(
                config, allow_full_run=args.allow_full_run, retry_errors=args.retry_errors
            )
            print(config.output_dir / "manifest.json")
            if not report["complete"]:
                parser.exit(2, "error: incomplete baseline; inspect inference_errors.jsonl\n")
        elif args.command == "prepare-sft":
            config = load_sft_config(args.config)
            if args.complete and args.english_only:
                raise ValueError("--complete and --english-only are mutually exclusive")
            if args.complete:
                from .sft_completion import complete_sft

                report = complete_sft(config)
            else:
                report = prepare_sft(config, english_only=args.english_only)
            print(config.output_dir / "manifest.json")
            if report["artifacts"]["issues"]["count"]:
                parser.exit(2, "error: overlength examples retained; inspect SFT issues.jsonl\n")
        elif args.command == "train":
            report = train_adapter(
                load_training_config(args.config),
                mode=args.mode,
                confirm_gpu=args.confirm_gpu_training,
                resume=args.resume,
                check_hardware=args.check_hardware,
            )
            print(json.dumps(report, indent=2, default=str))
            if args.mode == "dry-run" and not report["ready"]:
                parser.exit(2, "error: preflight blockers listed above; no training started.\n")
        elif args.command == "evaluate-comparison":
            config = load_comparison_config(args.config)
            report = evaluate_comparison(
                config, allow_full_run=args.allow_full_run, retry_errors=args.retry_errors
            )
            print(config.baseline.output_dir / "summary.json")
            if not report["complete"]:
                parser.exit(2, "error: incomplete inference retained in comparison denominators\n")
        elif args.command == "summarize-comparison":
            summarize_comparison_file(
                args.input,
                args.test_set,
                args.output_dir,
                seed=args.seed,
                bootstrap_samples=args.bootstrap_samples,
            )
            print(args.output_dir / "summary.json")
        elif args.command == "score":
            report = score_file(args.input, args.output_dir)
            print(args.output_dir / "metrics.json")
            if report["overall"]["invalid_records"]:
                parser.exit(2, "error: invalid scoring records retained in score_errors.jsonl\n")
        else:
            print(run(args.config, dry_run=args.dry_run))
    except (ValueError, OSError, yaml.YAMLError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0
