# Translation pipeline

This page describes the reusable translation CLI. The published 98-question evaluation set was imported from separately prepared drafts and remained provisionally validated; it did not use the mock or paid-API examples below. See the [research report](research_report.md) for the final review scope.

The pipeline translates **questions only** into `en`, `zh` (Simplified Chinese), `de`,
and `ja`. English is copied byte-for-byte from `original_question`. Rationale and gold
answers are never sent to a translation provider. A versioned instruction requires
translation only, no solving or hints, unchanged numeric notation and names, and
preservation of units, quantities, conditions, negations, comparisons, and meaning.

A prompt cannot guarantee semantic equivalence. All non-English API results start
as `needs_review` or `flagged`, never automatically approved. In particular, matching
numeric tokens does not establish correct units, names, number words, logical
relations, or mathematical meaning. Bilingual human review is required.

## Offline run and resume

The prepared 200-question training manifest must exist first (see the main README).
From the repository root:

```sh
source .venv/bin/activate
pip install -e '.[dev]'
python -m multilingual_llm_reasoning translate --config configs/translate_mock.yaml --estimate-only
python -m multilingual_llm_reasoning translate --config configs/translate_mock.yaml
# Same command resumes/reuses the local cache; it does not repeat successful requests.
python -m multilingual_llm_reasoning translate --config configs/translate_mock.yaml
pytest
ruff check .
```

Default selection is the prepared **100 original test questions**. The mock produces
four rows per original, with `[MOCK <language>]` English copies for non-English rows.
These are deliberately **not translations** and always carry `mock_not_translation`.
They validate coverage, provenance, persistence, and review workflows at zero API cost.

Set `splits: [train, validation, test]` to include the prepared training subset and
validation holdout. Train/test use their subset views; validation uses its canonical
holdout. To use 2,000 training originals, point `manifest_path` at that preparation run.
Use a new `output_dir` when changing selection, model/provider, prompt, token cap,
prices, or upstream manifest. Ingestion verifies file hashes, subset membership,
original-question IDs, and canonical split disjointness before translation. Translation
never assigns new splits: every language inherits its original ID and split.

## Configure API access and estimate costs

The provided API adapter uses the official [OpenAI Responses API](https://developers.openai.com/api/reference/python/resources/responses/methods/create).
It sends only the source question and translation instructions, sets `store=False`,
disables SDK retries, and does not request tools. Provider calls use a finite timeout.
The adapter always uses the official endpoint, ignoring `OPENAI_BASE_URL` overrides.

1. Install SDK support: `pip install -e '.[dev,translation]'`.
2. Copy/edit `configs/translate_api.yaml`. Replace `REPLACE_WITH_MODEL_SNAPSHOT` with
   an available text model, preferably an immutable snapshot. Enter its **current
   standard input and output USD prices per million tokens** from the
   [official pricing page](https://developers.openai.com/api/docs/pricing). Prices are
   intentionally `null` in the template, so no stale or invented rate enables a run.
3. Set `OPENAI_API_KEY` in your local environment or secret manager. `.env.example`
   documents the name; `.env` files are ignored but are **not automatically loaded**.
   Never put a real key in YAML, source code, review files, or Git. Estimation and
   review-only commands need no credentials.
4. Review `max_requests`, `max_cost_usd`, `max_attempts`, and `max_output_tokens`.
   Limits are cumulative for this output directory, not reset each time you resume.
5. Estimate entirely offline:

```sh
python -m multilingual_llm_reasoning translate --config configs/translate_api.yaml --estimate-only
```

For 100 originals there are **300 first-attempt requests** (English is free).
The estimate prints a cold-cache input allowance, maximum output tokens, and costs
at your supplied rates. Input allowance uses UTF-8 bytes plus protocol overhead; it
is deliberately conservative and **not an exact model token count or a billing quote**.
Output allowance uses the configured cap. The calculation is:

```text
USD allowance = (input allowance × input USD/million
                 + output token cap × output USD/million) / 1,000,000
```

The report also shows an all-retries allowance. With three attempts, at most 900
provider requests could be attempted for the test subset, subject to the configured
request and cost limits. Actual successful usage comes from the provider response;
cached-input discounts are ignored conservatively. Provider billing is authoritative.
Wrong or outdated prices and token-count estimates can make local estimates differ
from your bill; also configure account-side spending controls where available.

**No paid requests were executed while implementing or testing this pipeline.**
Only after you choose to authorize a paid run, use:

```sh
python -m multilingual_llm_reasoning translate --config configs/translate_api.yaml --allow-paid
```

Without `--allow-paid`, an API run is refused even if credentials exist. The sample
model placeholder and missing prices also block paid execution. `--estimate-only`
and `--review-only` never construct an API client or send any requests.

## Cache, limits, and failure behavior

Each output directory contains a SQLite cache and request ledger protected by an
exclusive process lock. Cache keys include the exact source question, original ID,
target language, provider/model/adapter version, prompt content/version, and output
cap. Source manifests and translation settings are fingerprinted to reject incompatible
resumes. Only successful responses are reused; suspicious/empty responses are retained
for review rather than automatically spending more money to regenerate them.

Every attempt reserves estimated cost **before** sending a request. Successful usage
replaces that reservation; failed or interrupted calls conservatively retain it because
a timeout does not prove the provider performed no billable work. Retries and pending
attempts count toward the durable request cap. An unexpectedly high reported charge
halts further calls for inspection. A crash between remote success and local persistence
can cause a retry to duplicate remote work; exactly-once remote execution is not claimed.
Both attempts remain budgeted. Backoff is bounded exponential backoff with deterministic
jitter. Only connection/timeouts and HTTP 408/409/429/5xx are retried. SDK retries are off.

Rerun the same command to resume. You can raise cumulative cost/request limits or
`max_attempts` without invalidating successful translations. Exhausted transient failures
need a higher attempt limit; permanent failures need investigation and a new run.
Do not delete the ledger to reset budgets. Archive the directory if starting a new run.
Only one process may write a given output directory at once.

Incomplete coverage or a reached limit exits the CLI with code 2 and exports available
responses, missing-translation rows, the request ledger, and QA report. Interrupted
processes may not finish JSONL exports; rerunning reconstructs them from SQLite.
`complete: true` means all expected raw responses exist, **not** that they are valid
translations. `research_ready` stays false for mock runs, missing variants, and any
unreviewed/rejected non-English variant.

## Saved artifacts

Default offline output: `data/translations/gsm8k-test-mock/` (Git-ignored).

| Artifact | Purpose |
| --- | --- |
| `translations.sqlite3` | Persistent raw response cache, requests/reservations, append-only reviews |
| `raw_translations.jsonl` | Immutable response contents exported from cache; review never edits these |
| `translations.jsonl` | Current text and QA status after applying the latest manual review event |
| `request_ledger.jsonl` | Attempts, timestamps, state, accounted cost, safe error code |
| `review_events.jsonl` | Append-only imported human decisions/corrections |
| `human_review.csv` | Seeded sample of 25 originals in all four languages, plus every flagged case |
| `qa_report.json` | Coverage, flags/status counts, requests/cost, completeness/readiness |
| `manifest.json` | Input manifest hash, config, provenance and hashes of exports |

Raw rows include source text, translated text, language, original ID, split/source
split, provider, requested model, returned model/version, adapter version, prompt
version/hash, timestamp (UTC), provider response ID/status, usage, raw-text hash,
and QA status/flags. English uses identity provenance. Missing responses appear in
effective exports as explicit `translation_error` rows; the ledger gives error details.
Gold answers accompany effective records for later scoring but never enter API inputs.
Downstream prompting must use only `translated_text`, not entire serialized records.

QA compares **multisets** of numeric tokens (including repeated numbers), flags missing
or suspiciously short text, unchanged English, missing Chinese/Japanese script, selected
solution markers, incomplete responses, and mock output. These are conservative
heuristics: both false positives and undetected errors are possible. German language
identity and semantic equivalence are not automatically established.

## Human review and corrections

The sample is 25 distinct originals (configurable from 20–30), with all four variants
for each, plus every flagged case outside the sample. Small test fixtures use all
available originals. `sampled=True` distinguishes the sample from extra flagged cases.

Copy `human_review.csv` **outside the generated output files** before editing; exports
are regenerated on resume. For each non-English row reviewed, fill:

- `reviewer`, timezone-aware ISO `reviewed_at`, and nonempty `notes`;
- `decision`: `approved` or `rejected` (blank rows are not imported);
- optional `corrected_text`: the entire corrected question, never a solution.

Check every quantity (including number words), unit, currency, named entity,
comparison, negation, condition, relation, and the final question. Confirm that nothing
was omitted, added, solved, or converted. If the numeric heuristic flags a legitimate
form, explain why in `notes`; an explicit human approval can override the flag, which
remains visible. English cannot be edited. Spreadsheet formula-like text is prefixed
with an apostrophe in the CSV for safe viewing; raw JSONL retains the exact source.

Apply the copied, edited CSV without any provider calls:

```sh
python -m multilingual_llm_reasoning translate --config configs/translate_mock.yaml \
  --review-only --corrections /absolute/path/to/reviewed.csv
```

Use the API config instead for reviews of real API output. JSONL correction events are
also supported, with exactly these fields per line:

```json
{"original_problem_id":"<copy ID>","language":"de","raw_translation_sha256":"<copy raw hash>","corrected_text":"<full corrected question or null>","decision":"approved","reviewer":"<name>","reviewed_at":"2026-10-17T10:00:00+00:00","notes":"<what was checked or changed>"}
```

Use JSON `null` (not the string `"null"`) to retain the raw text. Each event is a full
review decision relative to the raw response, not a patch to a prior correction.
Imports validate the entire file before applying any event, verify the raw-text hash,
reject unknown IDs and English edits, and deduplicate exact event re-imports. A new
event appends history; the latest imported event for that ID/language is applied.
Raw cache entries, raw timestamps, original IDs, split boundaries, and source text
remain unchanged. Corrected text is rechecked by QA. `review_events.jsonl` preserves
reviewer, time, notes, and decisions, including superseded events.

## Extending providers

Implement `TranslationProvider.translate` returning `TranslationResult`, and raise
`TranslationError` with a safe code and retryability. Register the config name and
factory in `translation_config.py` and `translation_pipeline.py`. The pipeline owns
retries, persistence, budgeting, QA, and review; adapters must not hide retries or send
answers/rationales. Tests inject fake providers directly and block network sockets.

Keep translated GSM8K artifacts out of Git and retain GSM8K attribution/license if
redistributing them; the README describes the upstream MIT requirements.
