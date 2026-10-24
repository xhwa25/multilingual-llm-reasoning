# Offline figures and matched error analysis

The published study's measured tables and uncertainty estimates are in the [final research report](research_report.md). Paths under `outputs/comparison-mock/` below illustrate the offline CLI only and are not research evidence.

Both commands read saved files. They never load a model, call an API or start training.
Install plotting support locally and run the complete fixture tests:

```sh
pip install -e '.[dev,analysis]'
pytest
ruff check .
```

Matplotlib is optional for the rest of the repository. Plot-rendering tests skip if
it is absent; install the analysis extra to run those tests. Error-analysis tests do
not require Matplotlib. Synthetic fixtures prohibit network access.

## Render publication figures

```sh
MPLCONFIGDIR="$PWD/.cache/matplotlib" python -m multilingual_llm_reasoning plot --metrics outputs/comparison-mock/summary.json --output-dir outputs/comparison-mock/figures
```

Use a fresh output directory; generated figures and human review work are never
overwritten. Replace the input path with a real comparison summary when available.
The renderer accepts the saved JSON schema from `evaluate-comparison` or
`summarize-comparison`, not arbitrary CSV column guesses.

The five figures, each exported as **300-dpi PNG and vector SVG**, are:

1. Grouped exact-match accuracy by language and model condition.
2. Adapter accuracy change relative to base, in percentage points.
3. English-minus-other-language accuracy gaps, in percentage points.
4. Fraction of original questions correct in all four languages.
5. Parse failures divided by expected questions, with the denominator explicit.

Colors use the Okabe–Ito palette and are supplemented by hatches, markers and explicit
labels. SVG labels remain selectable text. `figures.md` supplies descriptions and links
to figures; `source_metrics.json` preserves exact underlying values for accessible
inspection. `plot_manifest.json` records source hashes, Matplotlib version, exported-file
hashes, descriptions, loss-log provenance and captured rendering warnings.

Whiskers use **only saved intervals**. The current comparison output has paired
bootstrap intervals for adapter deltas, English gaps and consistency. Accuracy and
parse-failure charts explicitly state that no intervals were saved. The plotting
command does not manufacture intervals, substitute standard deviations, or treat a
missing interval as zero uncertainty. Missing/reversed/nonfinite intervals and invalid
or inconsistent denominators are rejected.

Mock inputs carry an explicit mock banner on every figure. Inputs without verified
research readiness are labeled unverified/incomplete. These labels cannot turn the
existing mock comparison into evidence that fine-tuning improved performance. For
100-question interpretation and bootstrap limitations, see [comparison methodology](comparison.md).

## Training/validation loss when logs exist

The command automatically discovers conventional full-run logs under
`<comparison-output-parent>/training/{english_lora,multilingual_lora}/full/training_log.jsonl`.
You can instead supply exact locations, including downloaded Colab logs:

```sh
python -m multilingual_llm_reasoning plot --metrics outputs/comparison-qwen/summary.json --output-dir outputs/comparison-qwen/figures-with-loss --training-log english_lora=outputs/training/english_lora/full/training_log.jsonl --training-log multilingual_lora=outputs/training/multilingual_lora/full/training_log.jsonl
```

When `loss` or `eval_loss` samples exist, a sixth PNG/SVG figure plots them against
optimizer step. Solid/circle lines indicate training; dashed/square lines indicate
validation. No smoothing, interpolation or invented confidence bounds are applied.
Backward step jumps start separate segments rather than connecting resumed sessions.
`initial_eval_loss` alone is not a training curve; if no regular train/validation samples
exist the manifest records why the loss chart was skipped. Explicit nonexistent log
paths are errors. No synthetic loss curves are included in a real evaluation report.

Training losses may be interval averages, while validation losses come from the shared
holdout. Different reporting schedules and resumed segments should be considered when
comparing curves. Training-log identity is supplied by the condition/path mapping;
inspect its saved run metadata before using third-party logs in a publication.

## Matched error-analysis report

```sh
python -m multilingual_llm_reasoning analyze --input outputs/comparison-mock/predictions.jsonl --test-set outputs/comparison-mock/selected_test.jsonl --output-dir outputs/comparison-mock/error-analysis
```

The command re-scores raw predictions and uses the same strict three-condition ×
four-language join as comparison metrics. Duplicates, missing/extra keys, mismatched
prompts/decoding or gold values fail explicitly. No inner join silently drops examples.
All matching cases are exported without sampling. Outputs are deterministic for the
same source files and seed; file hashes link the report to its inputs.

- `report.md` and `report.json`: event counts, affected original problem IDs, method,
  mock status and source hashes.
- `events.jsonl`: question-level correctness/answer disagreements across languages;
  correct→incorrect and incorrect→correct changes for both adapters vs. base and
  multilingual vs. English adapter; translation QA flags/status/review evidence.
- `evidence.jsonl`: selected condition/ID/language rows with original and translated
  questions, raw prediction, gold, parsed answer, correctness, parse/inference status,
  translation QA/review and model/adapter provenance.
- `human_review.csv`: the same evidence, linked by stable `review_id`, with blank
  `reviewer_category`, `reviewer_notes`, and `reviewer` fields.

The candidate categories are **translation ambiguity, comprehension, arithmetic,
multi-step logic, formatting/parser, and unknown**. They are options, not automated
labels or ground truth. Reviewers should read the source question, translation and
full raw output, then annotate their reasoning and uncertainty. Preserve raw JSON
alongside edited CSV files, and use bilingual review for translation judgments.

Disagreement includes either correctness differences or differing parsed answers/
parse statuses, so two incorrect answers can still be an informative disagreement.
Inference and parse failures can cause correctness flips without indicating changed
reasoning ability. `incorrect_or_unscored` is an objective selection reason, not a
subjective error category. All incorrect/unscored rows are included even without a
cross-language disagreement.

A reviewed translation with remaining QA flags is still surfaced. Missing QA status
also prompts review. Flags identify suspicions, not established translation mistakes;
unflagged translations may still be wrong, and numeric-token equality does not prove
semantic equivalence. Events overlap, so event counts should not be summed as counts
of unique errors. `affected_original_ids` and per-row IDs make this explicit.

## Verification artifacts

The saved mock comparison produces five evaluation figures and a review report; no
training curves exist for that run. A separate synthetic rendering fixture exercises
nonzero intervals and loss curves. These diagnostics are labeled mock and must not be
reported as research findings. The tests cover strict joins, matched flips, reviewed
translation flags, blank subjective categories, deterministic reports, PNG/SVG
rendering, missing intervals/logs, nonfinite values and CLI behavior.
