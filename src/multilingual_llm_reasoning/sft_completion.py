"""Complete partial paired data only after validating reviewed translations and identical A data."""

import json
import tempfile
from dataclasses import replace
from pathlib import Path

from filelock import FileLock

from .sft import _artifact, prepare_sft
from .sft_config import SFTConfig


def complete_sft(config: SFTConfig, **kwargs) -> dict:
    with FileLock(str(config.output_dir.parent / ".sft-completion.lock"), timeout=0):
        path = config.output_dir / "manifest.json"
        old = json.loads(path.read_text())
        if old.get("status") != "partial":
            raise ValueError("--complete requires an existing partial English-only export")
        for name in ("english_train", "validation"):
            _artifact(path, old["artifacts"][name])
        with tempfile.TemporaryDirectory(dir=config.output_dir.parent) as temporary:
            stage = Path(temporary) / "paired"
            report = prepare_sft(replace(config, output_dir=stage), **kwargs)
            if not report["training_ready"]:
                raise ValueError("Completion requires reviewed real translations and valid lengths")
            for name in ("english_train", "validation"):
                if report["artifacts"][name]["sha256"] != old["artifacts"][name]["sha256"]:
                    raise ValueError(
                        "Completion changed English/validation data; use original settings"
                    )
            report["config"]["output_dir"] = str(config.output_dir)
            # Existing English/validation bytes stay untouched. Publish manifest last.
            for name, artifact in report["artifacts"].items():
                if name not in ("english_train", "validation"):
                    (stage / artifact["path"]).replace(config.output_dir / artifact["path"])
            staged_manifest = stage / "manifest.json"
            staged_manifest.write_text(json.dumps(report, default=str, sort_keys=True) + "\n")
            staged_manifest.replace(path)
            return report
