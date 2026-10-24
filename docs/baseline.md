# Baseline inference and scoring

The target is [`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct).
The runner supports local Transformers, an optional local vLLM adapter, and an explicit
mock backend. It never uses paid inference APIs. Real backends download model/tokenizer
files on first use; weights and caches must remain outside Git. The model card lists a
Qwen Research license; consult it before using or redistributing model weights.

## Verified smoke test

From the repository root, with the prepared mock translation artifacts available:

```sh
source .venv/bin/activate
pip install -e '.[dev]'
python -m multilingual_llm_reasoning evaluate-baseline --config configs/baseline_mock.yaml
python -m multilingual_llm_reasoning score \
  --input outputs/baseline-smoke-mock/predictions.jsonl \
  --output-dir outputs/baseline-smoke-mock-scored
pytest
ruff check .
```

The default is **5 distinct official test originals × 4 languages = 20 rows**. The
same five originals are used in `en`, `zh`, `de`, and `ja`. Selection hashes the seed
and original ID, independent of input row ordering. The mock returns `FINAL_ANSWER: 0`
without reading gold answers. It identifies its actual checkpoint/revision as
`mock-baseline-v1`, with Qwen recorded only as `requested_model`.

This smoke test has been executed with the mock backend. The existing non-English
translations are mock placeholders, so these scores are **plumbing checks, not Qwen
results or cross-lingual research findings**. No model weights were downloaded and no
full benchmark was launched. Framework adapter tests use fake tokenizers/models and
block network access; real Transformers/vLLM execution still requires hardware validation.

`evaluate-baseline` resumes in place. `score` refuses to overwrite an existing output
directory; use a new scoring output path when rerunning the commands above.

## Real Transformers smoke run

Complete real translation and bilingual review first. The real config points to the
API translation artifact and refuses mock/unreviewed inputs by default. Then:

```sh
pip install -e '.[inference]'
HF_HOME="$PWD/.cache/huggingface" python -m multilingual_llm_reasoning evaluate-baseline \
  --config configs/baseline_qwen.yaml
```

Edit the YAML for your reviewed translation manifest and hardware. `device: auto`
selects CUDA GPU 0, then Apple MPS, then CPU. Explicit `cpu`, `mps`, `cuda`, and
`cuda:N` are supported; unavailable requested hardware is an error, not a silent
fallback. This adapter uses one device, not automatic multi-device sharding.

`dtype: auto` selects float32 on CPU, float16 on MPS, and bfloat16 on compatible CUDA
hardware (float16 otherwise). Explicit supported dtype choices are recorded. The CPU
adapter requires float32; MPS accepts float16/float32. Batch size defaults to 2 for
real smoke runs. If memory is insufficient, choose batch size 1 in a **new output
directory**, then rerun the five-question smoke test. Model load and generation errors
are retained for each affected variant; the program does not silently switch hardware,
quantization, decoding policy, or batch size.

A model branch/tag is resolved once to an immutable Hub commit before loading. Both
model and tokenizer use that exact revision, which is saved and reused on resume.
Pin the saved SHA in configurations for subsequent new runs. A complete cache hit
requires neither model loading nor Hub access. Resuming pending work checks the actual
model revision, template, device, dtype, and library versions against stored metadata.

For an exploratory real-model smoke test on unreviewed inputs, an explicit
`allow_unreviewed_translations: true` setting is available. Such outputs are labeled
as unsuitable for research interpretation. Do not use the mock-language copies for
language comparisons.

## Optional vLLM

Install vLLM separately in a compatible CUDA environment following the
[official installation guidance](https://docs.vllm.ai/en/stable/getting_started/installation/).
It is not installed by the default package or the Transformers extra. Then use
`configs/baseline_vllm.yaml` with the same reviewed input, revision, instruction, seed,
and decoding settings:

```sh
HF_HOME="$PWD/.cache/huggingface" python -m multilingual_llm_reasoning evaluate-baseline \
  --config configs/baseline_vllm.yaml
```

The adapter uses one visible CUDA GPU, `vllm.LLM.chat`, the model tokenizer's chat
template, greedy `SamplingParams(temperature=0)`, and the checkpoint's EOS IDs.
Select physical hardware with `CUDA_VISIBLE_DEVICES`; this adapter addresses visible
GPU 0. Missing vLLM or unavailable hardware produces explicit error rows. It does not
automatically fall back to Transformers or mock inference.

## Prompt and decoding policy

The system instruction is exactly the same for every language:

> Solve the problem and give a concise answer. End your response with a separate line
> in exactly this format: FINAL_ANSWER: &lt;number&gt;. Replace &lt;number&gt; with a single signed
> decimal number using a dot for the decimal point, without units or surrounding text.
> Do not write anything after that final line.

The user message contains **only the translated question**. No reference answer,
source English duplicate, rationale, language-specific hint, or few-shot demonstration
is appended. `prompt_template_version` is `baseline-final-answer-v1`.

“Language-neutral” here means the **same task instruction and format** across languages;
it does not mean the instruction is linguistically neutral. The instruction itself is
**English**, and the answer marker and decimal notation also favor a shared English/US
format. These can affect instruction adherence and parse failure rates differently
across languages, confounding pure reasoning comparisons. Report that limitation;
a later experiment could compare localized instructions as a separately controlled
condition. Equal decoding settings do not eliminate tokenization or training-data bias.

Transformers uses the checkpoint tokenizer's `apply_chat_template` with
`add_generation_prompt=True`, then tokenizes rendered chats with
`add_special_tokens=False`. It left-pads batches, keeps attention masks, and slices
generated output at the **padded input width** so prompt text is not scored. It does
not truncate questions. See [Hugging Face chat templates](https://huggingface.co/docs/transformers/main/chat_templating).

All languages use greedy decoding, one beam, repetition penalty 1, seed 42, and the
same `max_new_tokens: 128` cap in the sample configs. Sampling is disabled. Transformers
uses `temperature=None` because greedy mode ignores temperature and some versions reject
zero; the effective policy is greedy/temperature zero. vLLM uses literal temperature
zero. Checkpoint sampling defaults are replaced, and stop IDs come from the checkpoint.
The resolved chat-template text/hash and decoding policy are persisted with each row.

Greedy outputs are not guaranteed bitwise identical across devices, dtypes, kernels,
batch compositions, backends, or library versions. A fixed token cap can also constrain
languages differently. Preserve these settings and versions in research reports.

## Cache, resume, and scope guard

An exclusive process lock protects a SQLite ledger. Input artifact hashes and the full
run configuration fingerprint prevent accidental reuse after changing input, prompt,
model, revision, batch size, or decoding settings. Successful outputs—including malformed
answers—are cached. Parse failures are not regenerated automatically.

Each batch has pending attempt records committed before generation. On interruption,
rerun the same command: completed records are reused and pending/unstarted records are
processed. Earlier attempts remain in `inference_attempts.jsonl`. A crash after generation
but before persistence may repeat that local computation; no exactly-once claim is made.
Recorded inference errors are retained by default. To retry them after resolving the
cause, use the same configuration with `--retry-errors`:

```sh
python -m multilingual_llm_reasoning evaluate-baseline \
  --config configs/baseline_qwen.yaml --retry-errors
```

Input validation rejects missing language variants, changed English, differing gold
answers, ID/split leakage, missing text, or hash mismatches rather than quietly filtering
rows. Each selected variant gets an output row even when generation or model loading
fails. Unexpected extra backend outputs are saved in `backend_anomalies.jsonl`.

All sample configs select five originals. Setting `num_questions` above 5 is refused
unless the user also supplies `--allow-full-run`. No command here requests a full run.

## Artifacts and timing

Default mock output: `outputs/baseline-smoke-mock/` (Git-ignored).

| Artifact | Content |
| --- | --- |
| `predictions.jsonl` | One latest row per original ID/language, including explicit failures |
| `inference_attempts.jsonl` | Complete attempt history, including failed and interrupted attempts |
| `parse_failures.jsonl` | Generated answers whose final number could not be parsed |
| `inference_errors.jsonl` | Model/load failures or pending inference rows |
| `backend_anomalies.jsonl` | Unexpected extra/invalid backend outputs, retained for diagnosis |
| `metrics.json` | Overall/per-language accuracy, parse statuses and failure counts |
| `manifest.json` | Selected IDs, config, source hash, resolved backend, coverage, artifact hashes |
| `baseline.sqlite3` | Durable run metadata and attempt cache |

Each prediction row stores original ID, language, source/assigned split, requested
model/ref, actual checkpoint/revision, backend/version, device/dtype, exact messages,
instruction/template/hash/version, decoding settings, raw prediction, parsed answer,
gold answer, correctness, parse status/version, finish reason, generated token count,
timestamp, latency, and errors when present. Gold is retained only for scoring.

`latency_seconds` is **batch wall time**, shared by the rows of that batch; it includes
tokenization, transfer and generation, but excludes model loading. Transformers
synchronizes CUDA/MPS before timing completion. `amortized_latency_seconds` divides
the batch duration by its size. Neither field is an independently measured single-query
latency. Do not sum shared batch durations or treat them as per-language speed evidence.

## Exact scoring and parse failures

`score` recomputes from raw predictions, ignoring stored `correct` or `parsed_answer`.
It writes `scored_predictions.jsonl`, `metrics.json`, `parse_failures.jsonl`, and
`score_errors.jsonl` into a new directory.

The versioned parser requires exactly one terminal `FINAL_ANSWER:` line. Explanatory
text before that line is allowed; numeric extraction from arbitrary prose is not.
A single number must occupy the entire suffix (surrounding whitespace is allowed).
Multiple markers, trailing explanations, units/currency labels, malformed grouping,
fractions, NaN, infinity, empty answers, and missing markers are explicit failures.

Numeric normalization is exact and uses decimal strings, never binary floating point
or tolerance. US comma grouping, dot decimals, signs (including Unicode minus),
scientific notation, leading zeros, and surrounding whitespace are handled:
`+001,234.500` equals `1234.5`, and `-0.000` equals `0`. `1,23` is rejected rather than
interpreted as a decimal comma. `0.30000000000000001` does not equal `0.3`.

Primary accuracy is correct answers divided by **all rows**, including parse failures
and inference errors. Reports separately count parse failures, inference failures/pending
rows, invalid records, and each parse status. `accuracy_among_parsed` is secondary and
must not replace the primary denominator. Malformed JSONL lines, duplicate records,
and invalid gold answers are retained with explicit errors rather than discarded.
Use one baseline run per scoring invocation; do not merge models into one score file.

A run exits with code 2 on incomplete inference; parse failures alone do not fail the
CLI because they are valid evaluation outcomes. The scoring CLI exits 2 for invalid
records while still saving its reports. Blank/invalid input lines are also accounted for.
