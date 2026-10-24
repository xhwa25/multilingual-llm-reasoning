"""Translation tests use fixtures/fakes exclusively, including the API adapter contract."""

import csv
import hashlib
import json
import socket
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from multilingual_llm_reasoning import preparation, translation_pipeline
from multilingual_llm_reasoning.cli import main
from multilingual_llm_reasoning.preparation import PreparationConfig, prepare_data
from multilingual_llm_reasoning.translation import (
    LANGUAGES,
    OpenAITranslationProvider,
    TranslationError,
    TranslationResult,
)
from multilingual_llm_reasoning.translation_config import TranslationConfig, load_translation_config
from multilingual_llm_reasoning.translation_pipeline import estimate_translation, translate_dataset
from multilingual_llm_reasoning.translation_qa import (
    mathematical_translation_flags,
    qa_flags,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        pytest.fail("Translation tests must never make network requests")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TranslationConfig:
    source = {
        split: [
            {
                "question": f"Alice has {i + 1} kg of rice ({split}). How much rice?",
                "answer": f"SECRET_RATIONALE\n#### {i + 1}",
            }
            for i in range(35)
        ]
        for split in ("train", "test")
    }
    monkeypatch.setattr(preparation, "load_source", lambda revision: (source, "a" * 40))
    manifest = prepare_data(PreparationConfig(42, 5, 4, 30, "a" * 40, tmp_path / "prepared"))
    raw = yaml.safe_load((ROOT / "configs/translate_mock.yaml").read_text())
    raw.update(manifest_path=str(manifest), output_dir=str(tmp_path / "translated"))
    path = tmp_path / "translate.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_translation_config(path)


def records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


class FakeProvider:
    def __init__(self, failures: int = 0, *, retryable: bool = True) -> None:
        self.calls: list[dict[str, Any]] = []
        self.failures = failures
        self.retryable = retryable

    def translate(self, text: str, **kwargs: Any) -> TranslationResult:
        self.calls.append({"text": text, **kwargs})
        if len(self.calls) <= self.failures:
            raise TranslationError("fake_failure", retryable=self.retryable)
        prefix = {"zh": "问题", "de": "Frage", "ja": "質問"}[kwargs["language"]]
        return TranslationResult(f"{prefix}: {text}", "fake-snapshot", 10, 20, "fake-response")


def api_config(config: TranslationConfig) -> TranslationConfig:
    # Injected fake provider, NEVER an actual API request.
    return replace(
        config,
        provider="openai",
        model="fake-snapshot",
        max_cost_usd=Decimal("10"),
        input_usd_per_million=Decimal("1"),
        output_usd_per_million=Decimal("2"),
    )


def test_four_variants_provenance_cache_and_review(config: TranslationConfig) -> None:
    fake = FakeProvider()
    report = translate_dataset(config, provider=fake)
    assert report["complete"] is True
    assert report["research_ready"] is False
    assert report["request_attempts"] == 90
    assert len(fake.calls) == 90
    raw_path = config.output_dir / "raw_translations.jsonl"
    before = raw_path.read_bytes()
    raw = records(raw_path)
    assert len(raw) == 120
    assert len({(r["original_problem_id"], r["language"]) for r in raw}) == 120
    for row in raw:
        assert row["split"] == row["source_split"] == "test"
        assert row["timestamp"] and row["translation_provider_version"]
        assert row["translation_model_version"] and row["prompt_version"]
        assert (
            row["raw_translation_sha256"]
            == hashlib.sha256(row["translated_text"].encode()).hexdigest()
        )
        if row["language"] == "en":
            assert row["source_text"] == row["translated_text"]
            assert row["qa_status"] == "source_original"
        else:
            assert row["qa_status"] == "flagged"  # Config marks these mock, never approved.
    for call in fake.calls:
        assert "SECRET_RATIONALE" not in call["text"] + call["instructions"]
        assert "Never solve" in call["instructions"]
        assert "named entity" in call["instructions"]
    second = translate_dataset(config, provider=fake)
    assert second["request_attempts"] == 90
    assert len(fake.calls) == 90
    assert raw_path.read_bytes() == before
    with (config.output_dir / "human_review.csv").open(newline="") as stream:
        review = list(csv.DictReader(stream))
    sampled = {r["original_problem_id"] for r in review if r["sampled"] == "True"}
    assert len(sampled) == 25
    assert {r["language"] for r in review if r["sampled"] == "True"} == set(LANGUAGES)
    assert len(review) == 25 * 4 + 5 * 3  # All flagged cases outside sample included.
    manifest = json.loads((config.output_dir / "manifest.json").read_text())
    for artifact in manifest["artifacts"].values():
        assert (
            hashlib.sha256((config.output_dir / artifact["path"]).read_bytes()).hexdigest()
            == (artifact["sha256"])
        )


def test_cross_split_boundaries(config: TranslationConfig) -> None:
    report = translate_dataset(replace(config, splits=("train", "validation", "test")))
    assert report["splits"] == {"train": 4, "validation": 5, "test": 30}
    membership = {}
    for row in records(config.output_dir / "translations.jsonl"):
        membership.setdefault(row["original_problem_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in membership.values())


@pytest.mark.parametrize(
    "source,target,flag",
    [
        ("Alice has 5 kg and 5 kg.", "Alice hat 5 kg.", "changed_numeric_tokens"),
        ("5 apples", "6 Äpfel", "changed_numeric_tokens"),
        ("5 apples", "", "missing_text"),
        ("A long condition about 5 apples and two other people", "5", "possibly_missing_text"),
        ("5 apples", "5 apples", "unchanged_english"),
    ],
)
def test_qa_flags(source: str, target: str, flag: str) -> None:
    assert flag in qa_flags(source, target, "de")


def test_numeric_equality_never_implies_approval(config: TranslationConfig) -> None:
    conf = api_config(config)
    fake = FakeProvider()
    report = translate_dataset(conf, allow_paid=True, provider=fake)
    assert report["qa_counts"] == {"source_original": 30, "needs_review": 90}
    assert report["research_ready"] is False
    assert qa_flags("Alice buys 5 apples", "Alice verkauft 5 Äpfel", "de") == []
    assert "provider_incomplete" in qa_flags("5", "5", "de", response_status="incomplete")


def test_retries_backoff_and_durable_request_limit(config: TranslationConfig) -> None:
    fake = FakeProvider(2)
    delays = []
    conf = replace(config, max_requests=2)
    report = translate_dataset(conf, provider=fake, sleep=delays.append)
    assert not report["complete"]
    assert report["request_attempts"] == 2
    assert "max_requests" in report["blocked_reason"]
    assert len(fake.calls) == 2
    again = translate_dataset(conf, provider=fake, sleep=delays.append)
    assert again["request_attempts"] == 2
    assert len(fake.calls) == 2
    resumed = translate_dataset(config, provider=fake, sleep=delays.append)
    assert resumed["complete"]
    assert resumed["request_attempts"] == 92
    assert len(delays) == 2 and 0.5 <= delays[0] <= 1 and 1 <= delays[1] <= 2


def test_cost_reservation_before_call(config: TranslationConfig) -> None:
    conf = replace(api_config(config), max_cost_usd=Decimal("0.000001"))
    fake = FakeProvider()
    report = translate_dataset(conf, allow_paid=True, provider=fake)
    assert len(fake.calls) == 0
    assert report["request_attempts"] == 0
    assert not report["complete"]
    assert "max_cost" in report["blocked_reason"]


def test_failures_keep_cost_reservation(config: TranslationConfig) -> None:
    fake = FakeProvider(1000)
    report = translate_dataset(
        replace(api_config(config), max_requests=2),
        allow_paid=True,
        provider=fake,
        sleep=lambda _: None,
    )
    assert report["request_attempts"] == 2
    assert Decimal(report["accounted_cost_usd"]) > 0
    ledger = records(config.output_dir / "request_ledger.jsonl")
    assert all(r["state"] == "retryable" for r in ledger)


def test_permanent_failure_and_retry_exhaustion(config: TranslationConfig) -> None:
    fake = FakeProvider(1000, retryable=False)
    report = translate_dataset(config, provider=fake)
    assert not report["complete"]
    assert len(fake.calls) == 90
    translate_dataset(config, provider=fake)
    assert len(fake.calls) == 90
    other = replace(config, output_dir=config.output_dir.parent / "exhausted", max_attempts=2)
    transient = FakeProvider(1000)
    report = translate_dataset(other, provider=transient, sleep=lambda _: None)
    assert len(transient.calls) == 180
    assert report["request_attempts"] == 180
    translate_dataset(other, provider=transient, sleep=lambda _: None)
    assert len(transient.calls) == 180


def test_crash_resume_keeps_prior_results_and_pending_charge(config: TranslationConfig) -> None:
    class Interrupted(FakeProvider):
        def translate(self, text: str, **kwargs: Any) -> TranslationResult:
            if len(self.calls) == 1:
                raise KeyboardInterrupt
            return super().translate(text, **kwargs)

    interrupted = Interrupted()
    conf = api_config(config)
    with pytest.raises(KeyboardInterrupt):
        translate_dataset(conf, allow_paid=True, provider=interrupted)
    resumed_provider = FakeProvider()
    report = translate_dataset(
        conf, allow_paid=True, provider=resumed_provider, sleep=lambda _: None
    )
    assert report["complete"]
    assert len(resumed_provider.calls) == 89
    assert report["request_attempts"] == 91
    ledger = records(conf.output_dir / "request_ledger.jsonl")
    assert ledger[1]["state"] == "pending"
    assert Decimal(ledger[1]["charge_usd"]) > 0


def test_estimation_and_paid_opt_in(
    config: TranslationConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        pytest.fail("API client should not be constructed")

    monkeypatch.setattr(translation_pipeline, "OpenAITranslationProvider", fail)
    conf = api_config(config)
    estimate = estimate_translation(conf)
    assert estimate["uncached_requests"] == 90
    assert estimate["variants"] == 120
    assert Decimal(estimate["first_attempt_allowance_usd"]) > 0
    assert not config.output_dir.exists()
    with pytest.raises(ValueError, match="allow-paid"):
        translate_dataset(conf)
    unpriced = replace(conf, input_usd_per_million=None)
    assert estimate_translation(unpriced)["first_attempt_allowance_usd"] is None
    with pytest.raises(ValueError, match="prices"):
        translate_dataset(unpriced, allow_paid=True)


def test_corrections_are_append_only_and_raw_unchanged(config: TranslationConfig) -> None:
    translate_dataset(config)
    before = (config.output_dir / "raw_translations.jsonl").read_bytes()
    row = next(
        r for r in records(config.output_dir / "raw_translations.jsonl") if r["language"] == "de"
    )
    event = {
        "original_problem_id": row["original_problem_id"],
        "language": "de",
        "raw_translation_sha256": row["raw_translation_sha256"],
        "corrected_text": "Alice hat Reis. Wie viel Reis?",
        "decision": "approved",
        "reviewer": "Test reviewer",
        "reviewed_at": "2026-10-17T10:00:00+00:00",
        "notes": "Synthetic review fixture",
    }
    path = config.output_dir.parent / "corrections.jsonl"
    path.write_text(json.dumps(event) + "\n")
    report = translate_dataset(config, corrections=path, review_only=True)
    assert report["request_attempts"] == 90
    assert (config.output_dir / "raw_translations.jsonl").read_bytes() == before
    effective = next(
        r
        for r in records(config.output_dir / "translations.jsonl")
        if r["original_problem_id"] == row["original_problem_id"] and r["language"] == "de"
    )
    assert effective["translated_text"] == event["corrected_text"]
    assert effective["qa_status"] == "human_approved"
    assert "changed_numeric_tokens" in effective["qa_flags"]
    translate_dataset(config, corrections=path, review_only=True)
    assert len(records(config.output_dir / "review_events.jsonl")) == 1
    path.write_text(json.dumps({**event, "decision": "rejected", "notes": "Reconsidered"}) + "\n")
    translate_dataset(config, corrections=path, review_only=True)
    assert len(records(config.output_dir / "review_events.jsonl")) == 2
    assert (config.output_dir / "raw_translations.jsonl").read_bytes() == before
    path.write_text(json.dumps({**event, "raw_translation_sha256": "stale"}) + "\n")
    with pytest.raises(ValueError, match="stale"):
        translate_dataset(config, corrections=path, review_only=True)


def test_csv_review_import(config: TranslationConfig) -> None:
    translate_dataset(config)
    path = config.output_dir / "human_review.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    row = next(r for r in rows if r["language"] == "ja")
    row.update(
        decision="rejected",
        reviewer="Reviewer",
        reviewed_at="2026-10-17T10:00:00Z",
        notes="Needs a real translation",
    )
    edited = config.output_dir.parent / "reviewed.csv"
    with edited.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    translate_dataset(config, corrections=edited, review_only=True)
    events = records(config.output_dir / "review_events.jsonl")
    assert len(events) == 1 and events[0]["decision"] == "rejected"


def test_changed_inputs_and_provider_reject_resume(config: TranslationConfig) -> None:
    translate_dataset(config)
    with pytest.raises(ValueError, match="changed"):
        translate_dataset(replace(config, max_output_tokens=2048))
    manifest = json.loads(config.manifest_path.read_text())
    test_path = config.manifest_path.parent / manifest["subsets"]["test"]["path"]
    test_path.write_text(test_path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        translate_dataset(config)


def test_api_adapter_contract_without_sdk_or_network(monkeypatch: pytest.MonkeyPatch) -> None:
    construction = []
    requests = []

    class FakeError(Exception):
        pass

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            construction.append(kwargs)
            self.responses = self

        def create(self, **kwargs: Any) -> SimpleNamespace:
            requests.append(kwargs)
            return SimpleNamespace(
                output_text="訳文",
                model="snapshot",
                id="response",
                status="completed",
                usage=SimpleNamespace(input_tokens=10, output_tokens=20),
            )

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(
            OpenAI=FakeClient,
            APIConnectionError=FakeError,
            APITimeoutError=FakeError,
            APIStatusError=FakeError,
        ),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "fake-unit-test-key")
    provider = OpenAITranslationProvider("snapshot", timeout_seconds=10)
    result = provider.translate(
        "question only", language="ja", instructions="translate only", max_output_tokens=128
    )
    assert result.text == "訳文" and result.output_tokens == 20
    assert construction[0]["max_retries"] == 0
    assert construction[0]["base_url"] == "https://api.openai.com/v1"
    assert requests == [
        {
            "model": "snapshot",
            "instructions": "translate only",
            "input": "question only",
            "max_output_tokens": 128,
            "store": False,
        }
    ]


def test_config_validation_and_cli(
    config: TranslationConfig, capsys: pytest.CaptureFixture[str]
) -> None:
    path = config.output_dir.parent / "translate.yaml"
    assert main(["translate", "--config", str(path), "--estimate-only"]) == 0
    assert json.loads(capsys.readouterr().out)["uncached_requests"] == 90
    assert main(["translate", "--config", str(path)]) == 0
    assert main(["translate", "--config", str(path), "--review-only"]) == 0
    raw = yaml.safe_load(path.read_text())
    for key, value in (
        ("schema_version", True),
        ("seed", -1),
        ("max_requests", True),
        ("max_cost_usd", "NaN"),
        ("splits", ["test", "test"]),
        ("review_sample_size", 5),
        ("prompt_version", "unknown"),
        ("timeout_seconds", 0),
        ("provider", "unknown"),
    ):
        path.write_text(yaml.safe_dump({**raw, key: value}))
        with pytest.raises(ValueError):
            load_translation_config(path)


def test_same_directory_lock_prevents_concurrent_requests(config: TranslationConfig) -> None:
    from filelock import FileLock

    config.output_dir.mkdir()
    fake = FakeProvider()
    with FileLock(str(config.output_dir / ".translation.lock")):
        with pytest.raises(ValueError, match="Another translation"):
            translate_dataset(config, provider=fake)
    assert not fake.calls


def test_review_only_never_constructs_api_client(
    config: TranslationConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    conf = api_config(config)
    translate_dataset(conf, allow_paid=True, provider=FakeProvider())

    def blocked(*args: object, **kwargs: object) -> None:
        pytest.fail("Review-only must never construct an API client")

    monkeypatch.setattr(translation_pipeline, "OpenAITranslationProvider", blocked)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert translate_dataset(conf, review_only=True)["complete"]


def test_price_allowance_overrun_halts_new_requests(config: TranslationConfig) -> None:
    class ExcessiveUsage(FakeProvider):
        def translate(self, text: str, **kwargs: Any) -> TranslationResult:
            self.calls.append(kwargs)
            return TranslationResult(text, "fake", input_tokens=1_000_000, output_tokens=0)

    fake = ExcessiveUsage()
    report = translate_dataset(api_config(config), allow_paid=True, provider=fake)
    assert len(fake.calls) == 1
    assert "exceeded" in report["blocked_reason"]
    assert report["accounted_cost_usd"] == "1"


def test_manual_english_change_rejected(config: TranslationConfig) -> None:
    translate_dataset(config)
    original = next(
        r for r in records(config.output_dir / "raw_translations.jsonl") if r["language"] == "en"
    )
    event = {
        "original_problem_id": original["original_problem_id"],
        "language": "en",
        "raw_translation_sha256": original["raw_translation_sha256"],
        "corrected_text": "changed",
        "decision": "approved",
        "reviewer": "Reviewer",
        "reviewed_at": "2026-10-17T10:00:00Z",
        "notes": "Not allowed",
    }
    path = config.output_dir.parent / "bad-review.jsonl"
    path.write_text(json.dumps(event) + "\n")
    with pytest.raises(ValueError, match="immutable English"):
        translate_dataset(config, corrections=path, review_only=True)


def test_translation_prompt_change_invalidates_cache(
    config: TranslationConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    translate_dataset(config)
    previous = translation_pipeline.translation_prompt
    monkeypatch.setattr(
        translation_pipeline, "translation_prompt", lambda lang: previous(lang) + " new"
    )
    with pytest.raises(ValueError, match="changed"):
        translate_dataset(config)


def test_csv_formula_text_is_escaped(config: TranslationConfig) -> None:
    class FormulaProvider(FakeProvider):
        def translate(self, text: str, **kwargs: Any) -> TranslationResult:
            return TranslationResult('=HYPERLINK("example")', "fake")

    translate_dataset(config, provider=FormulaProvider())
    with (config.output_dir / "human_review.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert all(r["raw_translated_text"].startswith("'=") for r in rows if r["language"] != "en")
    raw = records(config.output_dir / "raw_translations.jsonl")
    assert all(r["translated_text"].startswith("=") for r in raw if r["language"] != "en")


def test_local_transformers_provider_is_offline_and_free(config: TranslationConfig) -> None:
    local = replace(
        config,
        provider="local_transformers",
        model="facebook/nllb-200-distilled-600M",
        output_dir=config.output_dir.parent / "local",
    )
    fake = FakeProvider()
    report = translate_dataset(local, provider=fake)
    assert report["complete"] is True
    assert estimate_translation(local)["first_attempt_allowance_usd"] == "0"
    raw = records(local.output_dir / "raw_translations.jsonl")
    translated = next(row for row in raw if row["language"] == "zh")
    assert translated["translation_provider"] == "local_transformers"
    assert translated["prompt_version"] == "nllb-direct-v1"
    assert translated["prompt_sha256"] is None


@pytest.mark.parametrize(
    ("language", "text", "expected"),
    [
        ("zh", "百分比机会是多少?", "missing_rounding_instruction"),
        ("de", "Wie viel, auf die nächste ganze Zahl abgerundet?", "changed_rounding_direction"),
        ("ja", "いくらですか。", "missing_rounding_instruction"),
    ],
)
def test_math_translation_flags_rounding(language: str, text: str, expected: str) -> None:
    source = "What is the percentage chance, rounded to the nearest integer?"
    assert expected in mathematical_translation_flags(source, text, language)


def test_math_translation_flags_question_currency_and_quantities() -> None:
    source = "She watched each of 9 movies three times and spent $216. What is the average price?"
    german = "Sie sah 9 Filme dreimal und gab 216 Dollar aus."
    flags = mathematical_translation_flags(source, german, "de")
    assert "missing_final_question" in flags
    assert "possibly_truncated_output" not in flags
    assert "missing_numeric_values" not in flags
    assert "missing_currency_reference" not in flags
    assert "unclear_each_quantity_relation" in flags


@pytest.mark.parametrize(
    "ending", ["いくらですか。", "合計はいくらになりますか。", "何個ありますか？"]
)
def test_japanese_question_ka_punctuation_is_not_missing(ending: str) -> None:
    flags = mathematical_translation_flags("How many are there?", ending, "ja")
    assert "missing_final_question" not in flags


def test_japanese_statement_still_flags_missing_question() -> None:
    flags = mathematical_translation_flags("How many are there?", "合計は10個です。", "ja")
    assert "missing_final_question" in flags


def test_math_translation_flags_added_values_and_empty_output() -> None:
    source = "There are 9 films. What is the total?"
    flags = mathematical_translation_flags("What is 9?", "何 9 9?", "zh")
    assert "duplicated_or_added_numeric_values" in flags
    assert mathematical_translation_flags(source, "", "ja") == [
        "empty_translation",
        "possibly_truncated_output",
    ]


def test_numeric_semantics_normalize_spelled_and_cjk_numbers() -> None:
    from multilingual_llm_reasoning.translation_qa import mathematical_translation_flags

    source = "Each one of 9 films was watched three times. What is the price?"
    assert "missing_numeric_values" not in mathematical_translation_flags(
        source, "9 Filme wurden dreimal gesehen. Wie hoch ist der Preis?", "de"
    )
    assert "missing_numeric_values" not in mathematical_translation_flags(
        "There are 9 films, each watched three times. What is the cost?",
        "有9部电影，每部看三次。费用是多少?",
        "zh",
    )
    assert "duplicated_or_added_numeric_values" not in mathematical_translation_flags(
        "What is the chance?", "一个随机选择的卡片的概率是多少?", "zh"
    )
    assert "duplicated_or_added_numeric_values" not in mathematical_translation_flags(
        "What is the chance?", "百分比机会是多少?", "zh"
    )


def test_rounding_idiom_is_not_counted_as_added_four() -> None:
    from multilingual_llm_reasoning.translation_qa import mathematical_translation_flags

    source = "What is the percentage chance rounded to the nearest integer?"
    assert "duplicated_or_added_numeric_values" not in mathematical_translation_flags(
        source, "概率（四舍五入到最接近的整数）是多少？", "zh"
    )
    assert "duplicated_or_added_numeric_values" not in mathematical_translation_flags(
        source, "確率（四捨五入して最も近い整数）は何パーセントですか？", "ja"
    )


def test_round_of_calls_is_not_mistaken_for_rounding_instruction() -> None:
    source = "How many families are notified during the fourth round of calls?"
    flags = mathematical_translation_flags(source, "第四轮电话通知了多少个家庭？", "zh")
    assert "missing_rounding_instruction" not in flags
