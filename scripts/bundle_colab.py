"""Allowlisted project/data bundle for Colab; never includes credentials or model caches."""

import argparse
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


PHASE16I_FROZEN_FILES = (
    "outputs/phase16h-approved-sft-freeze-v1/dataset_freeze_manifest.json",
    "outputs/phase16h-approved-sft-freeze-v1/english_only_sft_final.jsonl",
    "outputs/phase16h-approved-sft-freeze-v1/multilingual_sft_final.jsonl",
    "outputs/phase16c-sft-candidate-v1/sft_export/validation.jsonl",
    "outputs/phase16i-training-preparation-v1/sft_training_manifest.json",
)
PHASE16L_EVALUATION_FILES = (
    "outputs/phase14i-lite-provisional-dataset-v1/manifest.json",
    "outputs/phase14i-lite-provisional-dataset-v1/variants.jsonl",
    "outputs/phase14i-lite-real-baseline-v1/results/manifest.json",
    "outputs/phase14i-lite-real-baseline-v1/results/predictions.jsonl",
)


def bundle(
    output: Path, *, phase16i_frozen: bool = False, phase16l_evaluation: bool = False
) -> None:
    roots = [
        ROOT / p
        for p in (
            "pyproject.toml",
            "README.md",
            "src",
            "configs",
            "docs",
            "notebooks",
            "scripts",
            "requirements",
            "tests",
            "data/fixtures",
            "data/GSM8K_LICENSE.txt",
            "data/sft/gsm8k-200",
        )
    ]
    if phase16i_frozen:
        missing = [name for name in PHASE16I_FROZEN_FILES if not (ROOT / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Phase 16I frozen bundle is incomplete: {missing}")
        roots.extend(ROOT / name for name in PHASE16I_FROZEN_FILES)
    if phase16l_evaluation:
        missing = [name for name in PHASE16L_EVALUATION_FILES if not (ROOT / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Phase 16L evaluation bundle is incomplete: {missing}")
        roots.extend(ROOT / name for name in PHASE16L_EVALUATION_FILES)
    if output.exists():
        raise ValueError("Bundle already exists; choose a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for root in roots:
            paths = [root] if root.is_file() else sorted(root.rglob("*"))
            for path in paths:
                if not path.is_file() or path.is_symlink():
                    continue
                if any(
                    p.startswith(".") or p == "__pycache__" for p in path.relative_to(ROOT).parts
                ):
                    continue
                if path.suffix in (".pyc", ".safetensors", ".pt", ".bin", ".pth"):
                    continue
                archive.add(
                    path, arcname=str(Path("multilingual-llm-reasoning") / path.relative_to(ROOT))
                )
    print(output.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/colab-project.tar.gz")
    parser.add_argument(
        "--phase16i-frozen",
        action="store_true",
        help="Include only the approved Phase 16H datasets and Phase 16I training inputs",
    )
    parser.add_argument(
        "--phase16l-evaluation",
        action="store_true",
        help="Include the fixed 98-question provisional test set and verified baseline outputs",
    )
    args = parser.parse_args()
    bundle(
        args.output,
        phase16i_frozen=args.phase16i_frozen,
        phase16l_evaluation=args.phase16l_evaluation,
    )
