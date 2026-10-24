# Paired base / English LoRA / multilingual LoRA evaluation

This is the evaluator's technical guide. The completed 98-question results and the exact final run configuration are documented in the [research report](research_report.md) and [Phase 16L guide](evaluation/phase16l_colab_evaluation.md). The mock and 100-question commands below are software examples, not the published experiment.

`evaluate-comparison` evaluates the three fixed IDs `base`, `english_lora`, and
`multilingual_lora` on exactly the same original GSM8K official-test IDs and en/zh/de/ja
variants. It reuses baseline chat formatting, greedy decoding, response parsing and
exact numeric-equivalence scoring. There are no condition-specific prompts or limits.
The identical system instruction is English, which remains a language-bias limitation.

## Commands

Offline plumbing, no model weights, keys, GPU or real adapters:

```sh
python -m multilingual_llm_reasoning evaluate-comparison --config configs/comparison_mock.yaml --allow-full-run
```

The provided mock config selects 100 originals and produces 1,200 predictions and a
12-row condition/language summary. All three mock conditions return `FINAL_ANSWER: 0`;
they do not simulate fine-tuning improvements. The test translations are placeholders.
Outputs are marked `mock: true`, `research_ready: false`. They are not research results.
Rerunning resumes cached inference and deterministically regenerates metrics. Use a
new `output_dir` when settings, translation bytes or adapter files change.

For real evaluation, use the training environment (already contains Transformers/PEFT)
and edit `configs/comparison.yaml` adapter paths and `configs/comparison_baseline.yaml`
test-translation path/device as needed. The example paths use the local full-training
adapter directories; Colab Drive outputs require pointing to their exported adapters.
Both adapters and reviewed **test** translations must actually exist. No mock fallback.

Start by copying these two configs, setting `num_questions: 5`, and choosing a separate
output directory. Run without `--allow-full-run`. After reviewing this smoke, the
100-question command is:

```sh
.venv-colab/bin/python -m multilingual_llm_reasoning evaluate-comparison --config configs/comparison.yaml --allow-full-run
```

This runs inference only. It does not train adapters. The published 98-question benchmark used the separate Phase 16L configuration and is complete; this 100-question example was not its execution command. Add `--retry-errors` to retry inference errors; all
attempts are retained. A malformed generated answer is a parse failure, not a reason
to regenerate until a correct answer appears.

Recompute CSV/JSON metrics locally without loading any models or adapter weights:

```sh
python -m multilingual_llm_reasoning summarize-comparison --input outputs/comparison-mock/predictions.jsonl --test-set outputs/comparison-mock/selected_test.jsonl --output-dir outputs/comparison-mock-rescored --seed 42 --bootstrap-samples 5000
```

Offline rescoring does not certify research readiness; inspect original run provenance.
Its output directory must be new. Malformed lines and missing/extra/duplicate joins
fail explicitly rather than silently removing difficult examples.

## Adapter/base controls and leakage audit

The real backend creates a fresh base model for each condition at the same immutable
Qwen revision and uses its tokenizer, not an adapter-supplied tokenizer. Base has no
PEFT adapters attached. Each adapter is loaded unmerged, explicitly selected by name,
and checked via PEFT model status: enabled, exactly one active adapter, no merged
adapters, and nonzero adapter layers. Gradients are disabled and eval mode is used.
See [PEFT model activation](https://huggingface.co/docs/peft/package_reference/peft_model).

Final adapters produced by the updated trainer include `evaluation_provenance.json`
with model/revision, condition, train/validation original IDs and training budget.
The evaluator verifies adapter configuration and hashes its weights/config/provenance.
Train/validation IDs must be disjoint from selected test IDs; both adapters must have
matched train IDs, validation IDs and budgets. A missing provenance file blocks
inference. Do not invent provenance for an older checkpoint: export it only from its
verified original SFT manifest and run settings, or use the updated training export.
Current support targets final adapter exports, not arbitrary intermediate checkpoints.

All three models are evaluated in the same full-model precision (default FP16), even
when their adapters were trained with QLoRA. This differs from 4-bit training but is
consistent across evaluation conditions. Loading the base in full precision requires
sufficient inference memory; training's 4-bit memory footprint is not an inference
memory guarantee. Models run sequentially and cached CUDA memory is released between
conditions. vLLM adapter comparison is not enabled in this implementation.

A run contract binds selected IDs, translation-manifest hash, adapter hashes and all
settings. Metadata verifies identical checkpoint/revision, template, device and dtype.
Joins additionally verify shared prompts, decoding and gold answers by original ID and
language. Neither validation questions nor their loss values enter test accuracy.

## Artifacts and denominators

- `selected_test.jsonl`: frozen expected ID/language/gold/question grid.
- `predictions.jsonl`: combined rows with condition IDs, raw prediction, parsed answer,
  correctness, parse status, latency, model revision, prompts and adapter audit hashes.
- Each condition directory: resumable SQLite cache, raw inference attempts, predictions,
  parse failures, inference errors, anomalous backend returns and baseline manifest.
- `comparison_contract.json`: input/settings/adapter provenance and immutable signature.
- `summary.csv` / `summary.json`: 3 × 4 language metrics and transparent counts.
- `contrasts.csv`: English adapter minus base, multilingual adapter minus base, and
  multilingual minus English adapter, for each language, macro accuracy and consistency.
- `english_gaps.csv`: English accuracy minus each other language, within each condition.
- `consistency.csv`: fraction of originals answered correctly in **all four** languages.

Accuracy is `n_correct / n_expected`; each language has denominator 100 in the configured
subset. Parse failures and inference failures/pending rows count incorrect. Primary
parse-failure rate is `n_parse_failures / n_expected`; the rate among completed outputs
is also reported, with its `n_completed` denominator. Inference failures/pending have
a separate count. Consistency divides by the number of original problems (100), not
400 language variants. Macro accuracy averages four language accuracies over the same
originals. All numeric differences are proportions; multiply by 100 for percentage
points. Base-minus-itself has delta and interval exactly zero by definition.

An incomplete run retains expected rows and denominators, returns CLI exit 2, and is
not research-ready. Its error-inclusive metrics diagnose the failed run; do not use
those differences to claim model quality changes. The untouched raw output remains
available even if scoring fails. Saved correctness flags are recomputed from raw text.

## Paired uncertainty with only 100 originals

The seeded percentile bootstrap resamples **original problem IDs**, with replacement,
5,000 times. Each draw carries all four languages and all three conditions together.
The same draws are used for every contrast. For language differences, each original
contributes its paired correctness difference; macro and all-languages-correct metrics
are computed per original before resampling. English-language gaps are also paired
within original. Linear 2.5th/97.5th percentiles give the reported 95% intervals.

At 100 originals, one changed answer moves a language's accuracy by one percentage
point. Intervals can be wide or degenerate (especially near zero/one or with identical
paired outcomes). They reflect question-sampling variation, conditional on these
translations, adapters and decoding settings; they do not cover translation errors,
training-seed variation or broader model uncertainty. Intervals are exploratory and
unadjusted for multiple comparisons. Select questions/settings before looking at
results; do not repeatedly tune on the test set. Consider larger held-out samples and
replicated training seeds before strong claims.

No automatic narrative claims that fine-tuning helped. Report effect sizes, paired
intervals, failure rates and the study's limitations together. The available mock run
provides no evidence about fine-tuning effectiveness.
