import json
from pathlib import Path

import pytest
import yaml

from multilingual_llm_reasoning.cli import main
from multilingual_llm_reasoning.config import load_config
from multilingual_llm_reasoning.datasets import load_problems
from multilingual_llm_reasoning.inference import MockReasoner
from multilingual_llm_reasoning.pipeline import run
from multilingual_llm_reasoning.scoring import exact_match
from multilingual_llm_reasoning.statistics import accuracy_by_language

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    raw = yaml.safe_load((ROOT / "configs/dry_run.yaml").read_text())
    raw["dataset_path"] = str(ROOT / "data/fixtures/synthetic.jsonl")
    raw["output_dir"] = "results"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_config_relative_paths(config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    config = load_config(config_path)
    assert config.output_dir == config_path.parent / "results"
    assert config.languages == ("en", "es")


@pytest.mark.parametrize(
    "key,value",
    [
        ("seed", True),
        ("schema_version", 2),
        ("languages", ["en", "en"]),
        ("languages", []),
        ("unknown", 1),
        ("lora", {"rank": 0, "alpha": 1, "dropout": 0.1}),
        ("lora", {"rank": 8, "alpha": 16, "dropout": 1.0}),
    ],
)
def test_invalid_config(config_path: Path, key: str, value: object) -> None:
    raw = yaml.safe_load(config_path.read_text())
    raw[key] = value
    config_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_config(config_path)


def test_dataset_alignment_and_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    row = {"id": "1", "language": "en", "prompt": "2+3", "answer": "5"}
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="same nonempty set"):
        load_problems(path, ("en", "es"))
    path.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate"):
        load_problems(path, ("en",))
    path.write_text("{broken json")
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        load_problems(path, ("en",))


def test_scoring_is_conservative() -> None:
    assert exact_match(" 5\n", "5")
    assert not exact_match("5.0", "5")
    assert not exact_match("The answer is 5", "5")
    assert accuracy_by_language([]) == {}
    assert MockReasoner().predict("unrelated", language="es") == "5"


def test_offline_pipeline(config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    output = run(config_path, dry_run=True)
    first = output.read_bytes()
    payload = json.loads(first)
    assert payload["dry_run"] is True
    assert len(payload["dataset_sha256"]) == 64
    assert payload["metrics"] == {
        "en": {"count": 2, "correct": 1, "accuracy": 0.5},
        "es": {"count": 2, "correct": 1, "accuracy": 0.5},
    }
    with pytest.raises(FileExistsError):
        run(config_path, dry_run=True)
    assert output.read_bytes() == first
    output.unlink()
    assert run(config_path, dry_run=True).read_bytes() == first


def test_live_run_fails_before_writing(config_path: Path) -> None:
    with pytest.raises(ValueError, match="not implemented"):
        run(config_path, dry_run=False)
    assert not (config_path.parent / "results").exists()


def test_cli(config_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "validate" in capsys.readouterr().out
    assert main(["validate", "--config", str(config_path)]) == 0
    assert main(["run", "--config", str(config_path), "--dry-run"]) == 0
    with pytest.raises(SystemExit) as exc:
        main(["run", "--config", str(config_path)])
    assert exc.value.code == 2
