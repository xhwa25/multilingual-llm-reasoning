# Phase 16L: Colab held-out evaluation handoff

This is a prepared workflow; no Phase 16L adapter inference or comparison results exist until the notebook is deliberately run. The fixed test set has 98 canonical GSM8K questions and 392 `en`/`zh`/`de`/`ja` variants. Its Phase 14I manifest has `research_ready: false`: translation QA was limited, and human approvals remain pending. All Phase 16L reports must retain that provisional status. The instruction wrapper is in English for every language, which may affect cross-language comparisons.

The base checkpoint is `Qwen/Qwen2.5-3B-Instruct` revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, FP16, batch size 1, greedy decoding (`do_sample: false`, temperature 0), 256 maximum new tokens, seed 42, prompt `baseline-final-answer-v1`, and parser `final-answer-v2`. The existing 392 real base responses were generated with MPS; the two LoRA conditions will use CUDA. The baseline-reuse audit checks the pinned revision, all exact variant IDs/text hashes, prompts, decoding, parser, and prediction-file hash. Cross-device floating-point behavior remains a limitation.

## Build and upload

On the Mac, from the project root:

```bash
python3 scripts/bundle_colab.py --phase16i-frozen --phase16l-evaluation --output outputs/phase16l-colab-upload-v3.tar.gz
```

This allowlisted archive includes the frozen SFT manifests/files for leakage verification, fixed provisional test manifest/variants, saved real baseline manifest/predictions, code, configs, notebook, and tests. It excludes model weights, adapters, credentials, and private caches. Upload the archive when the notebook prompts. The Phase 16K final adapters must already exist in `MyDrive/multilingual-llm-reasoning/phase16k-full-training-v1/{english,multilingual}/final/adapter/` with their `completion.json` files. The notebook verifies the 75-step completion markers and SHA-256 hashes before copying adapters into an isolated local evaluation directory. Adapter provenance must match the pinned base, both conditions must have the same training IDs/validation IDs and budget, and neither may overlap the held-out test IDs.

Open `notebooks/phase16l_colab_final_evaluation.ipynb` in a fresh GPU Colab runtime. Cells 1–7 verify the GPU, install the locked CUDA environment, audit the test grid and baseline, mount Drive, copy/verify both adapters, and optionally load each adapter on a fresh base. The adapter-load switch defaults to `False`; enable it to perform a no-inference load check before evaluation. The actual evaluation switches `RUN_ENGLISH_EVAL`, `RUN_MULTILINGUAL_EVAL`, and `RUN_FINALIZE` also default to `False`. Enable one at a time. No cloud resource is started by the project itself; Colab GPU usage must be initiated by the user.

The CLI equivalents inside the Colab project are:

```bash
.venv-colab/bin/python scripts/phase16l_evaluation_runner.py --stage english_lora --drive-root /content/drive/MyDrive/multilingual-llm-reasoning/phase16l-final-evaluation-v1 --confirm-evaluation
.venv-colab/bin/python scripts/phase16l_evaluation_runner.py --stage multilingual_lora --drive-root /content/drive/MyDrive/multilingual-llm-reasoning/phase16l-final-evaluation-v1 --confirm-evaluation
.venv-colab/bin/python scripts/phase16l_evaluation_runner.py --stage finalize --drive-root /content/drive/MyDrive/multilingual-llm-reasoning/phase16l-final-evaluation-v1 --confirm-evaluation
```

The runner refuses CPU/MPS fallback, incomplete adapter provenance, changed experiment contracts, and existing completion markers. Each adapter condition loads a fresh base in a separate process; PEFT activation requires exactly the intended unmerged adapter. The finalizer requires both 392-variant completion markers, reuses the saved base responses, strictly joins all 1,176 rows, and copies files to a new Drive `final/` directory with SHA-256 values in `reproducibility_manifest.json`.

For a Colab disconnect, recreate the same project and verified adapters in a new runtime, then rerun **only the unfinished** stage with `--resume-from-drive` added to its command. The runner restores an integrity-checked online SQLite backup into an empty local output directory. Completed variants remain cached under the same signature. Do not use `--resume-from-drive` in a runtime with nonempty local comparison output.

## Saved outputs and interpretation

During inference, the runner mirrors each condition's SQLite progress store under `phase16l-final-evaluation-v1/progress/` about every 30 seconds. A completed final run saves `summary.json`, `summary.csv`, `overall_accuracy.csv`, `parsing_failures.csv`, `contrasts.csv`, `mcnemar.csv`, `english_gaps.csv`, `consistency.csv`, `statistical_comparison.md`, `final_evaluation_report.md`, joined and per-condition prediction JSONL, the selected test grid, comparison contract, and a checksum manifest under `final/`. Prediction rows contain raw responses, parse outcomes, gold answers, correctness, model/condition, prompts, and decoding settings. Parse failures remain incorrect in the denominator of 98 per language and condition.

The primary contrast is multilingual LoRA minus English LoRA. Paired percentile bootstrap intervals use 5,000 shared resamples of **canonical question IDs**, carrying all four languages and all conditions together. Exact two-sided McNemar tests are per language, with one outcome per canonical ID; they are exploratory and unadjusted for multiple comparisons. The notebook can additionally run the existing offline `analyze` and `plot` functions after completion to generate question-level error reports and PNG/SVG figures. Its `RUN_OFFLINE_REPORTS` switch also defaults to `False`. No result should be interpreted as a confirmed training benefit without examining the actual intervals, paired discordances, translation risks, and 98-question sample size.

Validation on the Mac before handoff:

```bash
pytest -q
ruff check .
python3 -m multilingual_llm_reasoning --help
```
