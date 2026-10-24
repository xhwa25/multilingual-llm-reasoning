"""Shared guards for checks against private, Git-ignored experiment inputs."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def private_final_run_inputs() -> None:
    """Skip a fresh clone, but reject a partially restored private experiment."""
    required = (
        ROOT / "outputs/phase14i-lite-provisional-dataset-v1/manifest.json",
        ROOT / "outputs/phase14i-lite-real-baseline-v1/results/manifest.json",
        ROOT / "outputs/phase16c-sft-candidate-v1/sft_export/validation.jsonl",
        ROOT / "outputs/phase16h-approved-sft-freeze-v1/dataset_freeze_manifest.json",
        ROOT / "outputs/phase16h-approved-sft-freeze-v1/english_only_sft_final.jsonl",
        ROOT / "outputs/phase16h-approved-sft-freeze-v1/multilingual_sft_final.jsonl",
        ROOT / "outputs/phase16i-training-preparation-v1/sft_training_manifest.json",
    )
    present = [path for path in required if path.is_file()]
    if not present:
        pytest.skip("Private final-run inputs are not included in the Git repository")
    missing = [str(path) for path in required if not path.is_file()]
    assert not missing, f"Partially restored private final-run inputs: {missing}"
