"""No weights, GPU, network or model dependencies are needed by baseline tests."""

import hashlib
import json
import socket
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from multilingual_llm_reasoning import baseline_backends
from multilingual_llm_reasoning.baseline import evaluate_baseline
from multilingual_llm_reasoning.baseline_backends import (
    Generation,
    MockBaselineBackend,
    TransformersBackend,
    VLLMBackend,
    resolve_revision,
    select_device,
)
from multilingual_llm_reasoning.baseline_config import (
    INSTRUCTION,
    BaselineConfig,
    load_baseline_config,
    make_messages,
)
from multilingual_llm_reasoning.baseline_scoring import score_file
from multilingual_llm_reasoning.cli import main
from multilingual_llm_reasoning.preparation import original_problem_id
from multilingual_llm_reasoning.scoring import numerical_equivalence, parse_final_answer

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        pytest.fail("Baseline unit tests must never access the network")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.mark.parametrize(
    "value,canonical",
    [
        ("5", "5"),
        ("+005.00", "5"),
        ("1,234.500", "1234.5"),
        ("-1,234.50", "-1234.5"),
        ("−.50", "-0.5"),
        ("-0.000", "0"),
        ("5e-2", "0.05"),
        ("\t 3.0 \n", "3"),
        ("123456789012345678901234567890.123000", "123456789012345678901234567890.123"),
    ],
)
def test_exact_answer_parsing(value: str, canonical: str) -> None:
    parsed = parse_final_answer(f"Concise explanation with 17 and 18.\n  FINAL_ANSWER: {value}")
    assert parsed.status == "ok"
    assert parsed.answer == canonical
    assert numerical_equivalence(value, canonical)


@pytest.mark.parametrize(
    "text,status",
    [
        ("", "empty_output"),
        (None, "empty_output"),
        ("5", "missing_marker"),
        ("final_answer: 5", "missing_marker"),
        ("FINAL_ANSWER: 5\nFINAL_ANSWER: 5", "multiple_markers"),
        ("FINAL_ANSWER: 1,23", "invalid_number"),
        ("FINAL_ANSWER: 5 or 6", "invalid_number"),
        ("FINAL_ANSWER: 5\nexplanation", "invalid_number"),
        ("FINAL_ANSWER: $5", "invalid_number"),
        ("FINAL_ANSWER: 5..", "invalid_number"),
        ("FINAL_ANSWER: 3.5.", "invalid_number"),
        ("FINAL_ANSWER: 1/2", "invalid_number"),
        ("FINAL_ANSWER: NaN", "invalid_number"),
        ("FINAL_ANSWER:", "invalid_number"),
    ],
)
def test_malformed_answers_not_guessed(text: str | None, status: str) -> None:
    assert parse_final_answer(text).status == status
    assert parse_final_answer(text).answer is None


@pytest.mark.parametrize(
    "text,canonical",
    [
        ("FINAL_ANSWER: 180", "180"),
        ("FINAL_ANSWER: 180.", "180"),
        ("FINAL_ANSWER: -12", "-12"),
        ("FINAL_ANSWER: 3.5", "3.5"),
        ("FINAL_ANSWER: 1,250", "1250"),
        ("FINAL_ANSWER:   + 180  \n", None),
        ("FINAL_ANSWER:  180  \n", "180"),
        ("My FINAL_ANSWER: 5", "5"),
        ("FINAL_ANSWER: -12.", "-12"),
        ("FINAL_ANSWER: 1,250.", "1250"),
        ("Reasoning mentions 34. FINAL_ANSWER: 180.", "180"),
    ],
)
def test_versioned_final_answer_formats(text: str, canonical: str | None) -> None:
    parsed = parse_final_answer(text)
    if canonical is None:
        assert parsed.status == "invalid_number"
    else:
        assert parsed.status == "ok"
        assert parsed.answer == canonical


def test_exact_equivalence_not_approximate() -> None:
    assert numerical_equivalence("1,000.00", "+1000")
    assert not numerical_equivalence("0.30000000000000001", "0.3")
    assert not numerical_equivalence("-5", "5")
    with pytest.raises(ValueError):
        numerical_equivalence("1,00", "100")


@pytest.fixture
def config(tmp_path: Path) -> BaselineConfig:
    folder = tmp_path / "translations"
    folder.mkdir()
    rows = []
    for index in range(7):
        source = f"Alice has {index + 1} apples. How many apples?"
        for lang in ("en", "zh", "de", "ja"):
            rows.append(
                {
                    "original_problem_id": original_problem_id(source),
                    "language": lang,
                    "split": "test",
                    "source_split": "test",
                    "source_text": source,
                    "translated_text": source if lang == "en" else f"{lang}: {source}",
                    "gold_answer": "5",
                    "qa_status": "source_original" if lang == "en" else "human_approved",
                }
            )
    content = "".join(json.dumps(row) + "\n" for row in rows)
    (folder / "translations.jsonl").write_text(content)
    manifest = {
        "schema_version": 1,
        "complete": True,
        "mock": False,
        "artifacts": {
            "variants": {
                "path": "translations.jsonl",
                "count": len(rows),
                "sha256": hashlib.sha256(content.encode()).hexdigest(),
            }
        },
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    raw = yaml.safe_load((ROOT / "configs/baseline_mock.yaml").read_text())
    raw.update(translation_manifest=str(folder / "manifest.json"), output_dir=str(tmp_path / "out"))
    path = tmp_path / "baseline.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_baseline_config(path)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


class FakeBackend(MockBaselineBackend):
    def __init__(self, config: BaselineConfig, revision: str) -> None:
        super().__init__(config, revision)
        self.calls = []

    def generate(self, messages: list) -> list[Generation]:
        self.calls.append(messages)
        return [Generation("FINAL_ANSWER: +005.00", "eos", 5) for _ in messages]


def test_smoke_20_rows_and_cache(config: BaselineConfig) -> None:
    fake = FakeBackend(config, "mock-baseline-v1")
    report = evaluate_baseline(config, backend_factory=lambda *_: fake)
    assert report["complete"]
    assert report["expected_variants"] == 20
    assert report["variants_by_language"] == {"en": 5, "zh": 5, "de": 5, "ja": 5}
    assert report["metrics"]["overall"]["accuracy"] == 1
    rows = read_rows(config.output_dir / "predictions.jsonl")
    assert len(rows) == len({(r["original_problem_id"], r["language"]) for r in rows}) == 20
    assert len({r["original_problem_id"] for r in rows}) == 5
    for row in rows:
        assert row["model_checkpoint"] == "mock-baseline-v1"
        assert row["requested_model"] == "Qwen/Qwen2.5-3B-Instruct"
        assert row["instruction"] == INSTRUCTION
        assert row["correct"] and row["parsed_answer"] == "5" and row["parse_status"] == "ok"
        assert row["latency_seconds"] >= row["amortized_latency_seconds"] >= 0
        assert row["decoding"]["do_sample"] is False
        assert row["split"] == row["source_split"] == "test"
    for batch in fake.calls:
        assert len(batch) == 4
        for messages in batch:
            assert messages[0] == {"role": "system", "content": INSTRUCTION}
            assert set(messages[1]) == {"role", "content"}  # No gold/rationale fields.
    before = (config.output_dir / "predictions.jsonl").read_bytes()

    def blocked(*args: object) -> Any:
        pytest.fail("Cached resume must not load models or resolve Hub revisions")

    evaluate_baseline(config, backend_factory=blocked, revision_resolver=blocked)
    assert (config.output_dir / "predictions.jsonl").read_bytes() == before
    assert len(read_rows(config.output_dir / "inference_attempts.jsonl")) == 20
    for artifact in report["artifacts"].values():
        assert (
            hashlib.sha256((config.output_dir / artifact["path"]).read_bytes()).hexdigest()
            == artifact["sha256"]
        )


def test_parse_failure_is_kept_and_reported(config: BaselineConfig) -> None:
    class Malformed(FakeBackend):
        def generate(self, messages: list) -> list[Generation]:
            return [Generation("Maybe 5?", "length") for _ in messages]

    report = evaluate_baseline(config, backend_factory=Malformed)
    assert report["complete"]
    assert report["metrics"]["overall"]["parse_failures"] == 20
    assert report["metrics"]["overall"]["accuracy"] == 0
    assert len(read_rows(config.output_dir / "parse_failures.jsonl")) == 20
    assert all(
        r["prediction"] == "Maybe 5?" for r in read_rows(config.output_dir / "predictions.jsonl")
    )


def test_error_rows_and_explicit_retry(config: BaselineConfig) -> None:
    class Failed(FakeBackend):
        def generate(self, messages: list) -> list[Generation]:
            raise RuntimeError("simulated out of memory")

    report = evaluate_baseline(config, backend_factory=Failed)
    assert not report["complete"]
    assert report["metrics"]["overall"]["inference_failures_or_pending"] == 20
    assert len(read_rows(config.output_dir / "inference_errors.jsonl")) == 20
    fake = FakeBackend(config, "mock-baseline-v1")
    evaluate_baseline(config, backend_factory=lambda *_: fake)
    assert not fake.calls
    report = evaluate_baseline(config, backend_factory=lambda *_: fake, retry_errors=True)
    assert report["complete"] and len(fake.calls) == 5
    attempts = read_rows(config.output_dir / "inference_attempts.jsonl")
    assert len(attempts) == 40
    assert sum(r["inference_status"] == "error" for r in attempts) == 20


def test_initialization_failure_produces_all_error_rows(config: BaselineConfig) -> None:
    def fail(*args: object) -> Any:
        raise ImportError("mock missing dependency")

    report = evaluate_baseline(config, backend_factory=fail)
    assert not report["complete"]
    assert len(read_rows(config.output_dir / "predictions.jsonl")) == 20
    assert all(
        r["error"].startswith("backend_initialization")
        for r in read_rows(config.output_dir / "inference_errors.jsonl")
    )


def test_crash_resume(config: BaselineConfig) -> None:
    class Crash(FakeBackend):
        def generate(self, messages: list) -> list[Generation]:
            if self.calls:
                raise KeyboardInterrupt
            return super().generate(messages)

    with pytest.raises(KeyboardInterrupt):
        evaluate_baseline(config, backend_factory=Crash)
    fake = FakeBackend(config, "mock-baseline-v1")
    report = evaluate_baseline(config, backend_factory=lambda *_: fake)
    assert report["complete"]
    assert sum(len(batch) for batch in fake.calls) == 16
    attempts = read_rows(config.output_dir / "inference_attempts.jsonl")
    assert len(attempts) == 24 and sum(r["inference_status"] == "pending" for r in attempts) == 4


def test_backend_count_mismatch_never_drops_outputs(config: BaselineConfig) -> None:
    class Short(FakeBackend):
        def generate(self, messages: list) -> list[Generation]:
            return [Generation("FINAL_ANSWER: 5", "eos")]

    report = evaluate_baseline(config, backend_factory=Short)
    assert not report["complete"]
    assert report["metrics"]["overall"]["correct"] == 5
    assert report["metrics"]["overall"]["total"] == 20
    other = replace(config, output_dir=config.output_dir.parent / "extra")

    class Extra(FakeBackend):
        def generate(self, messages: list) -> list[Generation]:
            return super().generate(messages) + [Generation("Extra output", "eos")]

    evaluate_baseline(other, backend_factory=Extra)
    anomalies = read_rows(other.output_dir / "backend_anomalies.jsonl")
    assert len(anomalies) == 5 and anomalies[0]["outputs"][0]["prediction"] == "Extra output"


def test_full_run_and_changed_config_guards(config: BaselineConfig) -> None:
    with pytest.raises(ValueError, match="allow-full-run"):
        evaluate_baseline(replace(config, num_questions=6))
    evaluate_baseline(config)
    with pytest.raises(ValueError, match="changed"):
        evaluate_baseline(replace(config, max_new_tokens=64))
    with pytest.raises(ValueError, match="more distinct"):
        evaluate_baseline(replace(config, num_questions=100), allow_full_run=True)


def test_unreviewed_inputs_require_opt_in(config: BaselineConfig) -> None:
    raw = json.loads(config.translation_manifest.read_text())
    raw["mock"] = True
    config.translation_manifest.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="mock/unreviewed"):
        evaluate_baseline(replace(config, allow_unreviewed_translations=False))


def test_score_keeps_malformed_and_duplicate_rows(tmp_path: Path) -> None:
    base = {
        "original_problem_id": "one",
        "language": "en",
        "model_checkpoint": "test",
        "model_revision": "revision",
        "gold_answer": "5",
        "prediction": "FINAL_ANSWER: 5.0",
        "inference_status": "completed",
        "correct": False,
    }
    path = tmp_path / "raw.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(base),
                json.dumps(base),
                "not json",
                json.dumps({**base, "original_problem_id": "two", "prediction": "5"}),
            ]
        )
        + "\n"
    )
    report = score_file(path, tmp_path / "scored")
    assert report["overall"]["total"] == 4
    assert report["overall"]["correct"] == 1
    assert report["overall"]["invalid_records"] == 2
    assert report["overall"]["parse_failures"] == 1
    assert report["overall"]["accuracy"] == 0.25
    assert len(read_rows(tmp_path / "scored/scored_predictions.jsonl")) == 4
    with pytest.raises(FileExistsError):
        score_file(path, tmp_path / "scored")


def test_cli(config: BaselineConfig) -> None:
    path = config.output_dir.parent / "baseline.yaml"
    assert main(["evaluate-baseline", "--config", str(path)]) == 0
    assert (
        main(
            [
                "score",
                "--input",
                str(config.output_dir / "predictions.jsonl"),
                "--output-dir",
                str(config.output_dir.parent / "scored"),
            ]
        )
        == 0
    )


class FakeTensor:
    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, key: tuple) -> "FakeTensor":
        return FakeTensor([row[key[1]] for row in self.rows[key[0]]])

    def tolist(self) -> list[list[int]]:
        return self.rows


class FakeEncoding(dict):
    def to(self, device: str) -> "FakeEncoding":
        self.device = device
        return self


def fake_torch() -> SimpleNamespace:
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
            is_bf16_supported=lambda: False,
            set_device=lambda _: None,
        ),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        manual_seed=lambda _: None,
        inference_mode=nullcontext,
        float32="float32",
        float16="float16",
        bfloat16="bfloat16",
    )


def test_transformers_chat_padding_and_greedy_contract(
    config: BaselineConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {}

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 99
        bos_token_id = 1

        def get_chat_template(self) -> str:
            return "real-checkpoint-template"

        def apply_chat_template(self, messages: list, **kwargs: Any) -> str:
            calls.setdefault("template", []).append((messages, kwargs))
            return "rendered"

        def __call__(self, prompts: list, **kwargs: Any) -> FakeEncoding:
            calls["tokenize"] = kwargs
            return FakeEncoding(
                input_ids=FakeTensor([[0, 1, 2], [1, 2, 3]]),
                attention_mask=FakeTensor([[0, 1, 1], [1, 1, 1]]),
            )

        def batch_decode(self, tokens: FakeTensor, **kwargs: Any) -> list[str]:
            calls["decoded_tokens"] = tokens.tolist()
            return ["FINAL_ANSWER: 5", "FINAL_ANSWER: -2"]

    class Model:
        generation_config = SimpleNamespace(eos_token_id=[99, 100])
        config = SimpleNamespace(max_position_embeddings=1024)

        def to(self, device: str) -> "Model":
            calls["device"] = device
            return self

        def eval(self) -> None:
            calls["eval"] = True

        def generate(self, **kwargs: Any) -> FakeTensor:
            calls["generate"] = kwargs
            return FakeTensor([[0, 1, 2, 7, 99, 0], [1, 2, 3, 8, 9, 100]])

    def tokenizer_load(*args: Any, **kwargs: Any) -> Tokenizer:
        calls["tokenizer_load"] = kwargs
        return Tokenizer()

    def model_load(*args: Any, **kwargs: Any) -> Model:
        calls["model_load"] = kwargs
        return Model()

    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_load),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_load),
            GenerationConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
    )
    monkeypatch.setattr(baseline_backends, "version", lambda _: "test-version")
    backend = TransformersBackend(replace(config, backend="transformers"), "a" * 40)
    messages = [make_messages("question one"), make_messages("question two")]
    outputs = backend.generate(messages)
    assert calls["tokenizer_load"]["padding_side"] == "left"
    assert calls["tokenizer_load"]["revision"] == calls["model_load"]["revision"] == "a" * 40
    assert all(
        kwargs == {"tokenize": False, "add_generation_prompt": True}
        for _, kwargs in calls["template"]
    )
    assert calls["tokenize"]["add_special_tokens"] is False
    assert calls["tokenize"]["truncation"] is False
    assert calls["decoded_tokens"] == [[7, 99, 0], [8, 9, 100]]
    generation = calls["generate"]["generation_config"]
    assert generation.do_sample is False and generation.temperature is None
    assert generation.max_new_tokens == 128 and generation.num_beams == 1
    assert outputs[0].generated_tokens == 2 and outputs[1].generated_tokens == 3
    assert calls["device"] == "cpu" and backend.metadata["dtype"] == "float32"


def test_vllm_chat_contract(config: BaselineConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    class LLM:
        def __init__(self, **kwargs: Any) -> None:
            calls["init"] = kwargs

        def get_tokenizer(self) -> Any:
            return SimpleNamespace(get_chat_template=lambda: "checkpoint-template")

        def chat(self, messages: list, **kwargs: Any) -> list:
            calls["chat"] = kwargs
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            text="FINAL_ANSWER: 5", finish_reason="stop", token_ids=[1, 2]
                        )
                    ]
                )
                for _ in messages
            ]

    torch = fake_torch()
    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = lambda: 1
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(LLM=LLM, SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs)),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            GenerationConfig=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: SimpleNamespace(eos_token_id=[99, 100])
            )
        ),
    )
    monkeypatch.setattr(baseline_backends, "version", lambda _: "test-version")
    backend = VLLMBackend(replace(config, backend="vllm", device="cuda:0"), "a" * 40)
    assert len(backend.generate([make_messages("one"), make_messages("two")])) == 2
    assert calls["init"]["revision"] == calls["init"]["tokenizer_revision"] == "a" * 40
    assert calls["init"]["generation_config"] == "vllm"
    assert calls["chat"]["add_generation_prompt"] is True
    assert backend.sampling.temperature == 0 and backend.sampling.max_tokens == 128
    assert backend.sampling.stop_token_ids == [99, 100]


def test_device_and_revision_resolution(
    config: BaselineConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = fake_torch()
    assert select_device(torch, "auto") == "cpu"
    with pytest.raises(ValueError, match="CUDA"):
        select_device(torch, "cuda:0")
    with pytest.raises(ValueError, match="MPS"):
        select_device(torch, "mps")

    class API:
        def model_info(self, model: str, *, revision: str) -> Any:
            assert model == config.model and revision == "main"
            return SimpleNamespace(sha="b" * 40)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=API))
    assert resolve_revision(replace(config, backend="transformers")) == "b" * 40
    assert resolve_revision(replace(config, backend="transformers", revision="a" * 40)) == "a" * 40


def test_config_validation(config: BaselineConfig) -> None:
    path = config.output_dir.parent / "baseline.yaml"
    raw = yaml.safe_load(path.read_text())
    for key, value in (
        ("temperature", 0.1),
        ("do_sample", True),
        ("batch_size", 0),
        ("num_questions", True),
        ("device", "tpu"),
        ("dtype", "int8"),
        ("model", "other"),
        ("prompt_version", "other"),
    ):
        path.write_text(yaml.safe_dump({**raw, key: value}))
        with pytest.raises(ValueError):
            load_baseline_config(path)


def test_invalid_backend_payload_is_preserved(config: BaselineConfig) -> None:
    class Invalid(FakeBackend):
        def generate(self, messages: list) -> list:
            return [{"unexpected_prediction": "must be retained"} for _ in messages]

    report = evaluate_baseline(config, backend_factory=Invalid)
    assert not report["complete"]
    assert len(read_rows(config.output_dir / "predictions.jsonl")) == 20
    anomalies = read_rows(config.output_dir / "backend_anomalies.jsonl")
    assert len(anomalies) == 20
    assert all("must be retained" in row["output"]["unparsed_output"] for row in anomalies)


def test_split_leakage_and_missing_language_are_explicit_errors(config: BaselineConfig) -> None:
    manifest = json.loads(config.translation_manifest.read_text())
    path = config.translation_manifest.parent / "translations.jsonl"
    rows = read_rows(path)
    original_rows = [dict(row) for row in rows]
    for mode in ("leakage", "missing_language"):
        rows = [dict(row) for row in original_rows]
        if mode == "leakage":
            rows[0]["split"] = "train"
        else:
            rows.pop(0)
        content = "".join(json.dumps(row) + "\n" for row in rows)
        path.write_text(content)
        manifest["artifacts"]["variants"].update(
            count=len(rows), sha256=hashlib.sha256(content.encode()).hexdigest()
        )
        config.translation_manifest.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="multiple splits|en, zh, de and ja"):
            evaluate_baseline(config)
        assert not config.output_dir.exists()


def test_english_scope_preserves_multilingual_review_gate(config):
    from multilingual_llm_reasoning.baseline import load_baseline_inputs

    path = config.translation_manifest.parent / "translations.jsonl"
    rows = read_rows(path)
    for row in rows:
        if row["language"] != "en":
            row["qa_status"] = "needs_review"
    content = "".join(json.dumps(row) + "\n" for row in rows)
    path.write_text(content)
    manifest = json.loads(config.translation_manifest.read_text())
    manifest["artifacts"]["variants"]["sha256"] = hashlib.sha256(content.encode()).hexdigest()
    config.translation_manifest.write_text(json.dumps(manifest))
    strict = replace(config, allow_unreviewed_translations=False)
    selected, _, reviewed = load_baseline_inputs(replace(strict, languages=("en",)))
    assert reviewed and len(selected) == 5
    assert {row["language"] for row in selected} == {"en"}
    with pytest.raises(ValueError, match="mock/unreviewed"):
        load_baseline_inputs(strict)
    rows[0]["translated_text"] = "Changed English"
    content = "".join(json.dumps(row) + "\n" for row in rows)
    path.write_text(content)
    manifest["artifacts"]["variants"]["sha256"] = hashlib.sha256(content.encode()).hexdigest()
    config.translation_manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unmodified"):
        load_baseline_inputs(replace(strict, languages=("en",)))


def test_comparison_refuses_english_only_scope(config):
    from multilingual_llm_reasoning.comparison import ComparisonConfig, evaluate_comparison

    comparison = ComparisonConfig(replace(config, languages=("en",)), None, None, 42, 100)
    with pytest.raises(ValueError, match="all four languages"):
        evaluate_comparison(comparison)


def test_default_language_scope_preserves_legacy_cache_contract(config):
    from multilingual_llm_reasoning.baseline_config import baseline_config_dict

    assert "languages" not in baseline_config_dict(config)
    assert baseline_config_dict(replace(config, languages=("en",)))["languages"] == ("en",)
