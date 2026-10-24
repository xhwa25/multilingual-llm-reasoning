"""Offline smoke pipeline with provenance and deterministic artifacts."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .config import load_config
from .datasets import load_problems
from .inference import MockReasoner
from .logging import configure_logging
from .scoring import ScoredPrediction, exact_match
from .statistics import accuracy_by_language


def run(config_path: Path, *, dry_run: bool) -> Path:
    if not dry_run:
        raise ValueError("Live experiments are not implemented; use --dry-run")
    config = load_config(config_path)
    if config.model != "mock":
        raise ValueError("Dry run requires model: mock")
    problems = load_problems(config.dataset_path, config.languages)
    model = MockReasoner()
    rows = []
    for problem in problems:
        prediction = model.predict(problem.prompt, language=problem.language)
        rows.append(
            ScoredPrediction(
                problem.id, problem.language, prediction, exact_match(prediction, problem.answer)
            )
        )
    payload = {
        "schema_version": 1,
        "package_version": __version__,
        "dry_run": True,
        "notice": "Synthetic smoke results; not evidence of model reasoning ability.",
        "config": {
            **asdict(config),
            "dataset_path": str(config.dataset_path),
            "output_dir": str(config.output_dir),
        },
        "dataset_sha256": hashlib.sha256(config.dataset_path.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "predictions": [asdict(row) for row in rows],
        "metrics": accuracy_by_language(rows),
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output = config.output_dir / "results.json"
    # Refuse overwrite so previous research artifacts cannot be silently lost.
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    configure_logging().info("dry_run_completed")
    return output
