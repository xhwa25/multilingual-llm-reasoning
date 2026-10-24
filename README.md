# Multilingual LLM Reasoning

**Research question:** How does question language affect a language model's accuracy on the same mathematical word problems, and what changes after English-only or multilingual-question fine-tuning?

This completed personal ML/NLP project compares the pinned `Qwen/Qwen2.5-3B-Instruct` base checkpoint with two independently trained LoRA adapters on **98 held-out GSM8K problems × four languages**. Its strongest result is negative: both adapters scored below the untouched base. The multilingual adapter scored 3.3 percentage points above the English-only adapter overall, but the paired 95% interval includes zero. The translated test set had limited QA, so the findings are **exploratory**, not a claim about language groups or general reasoning ability.

## Results

Every condition answered the same 392 variants (98 each in English, Chinese, German, and Japanese). Accuracy uses **all expected variants** as its denominator; malformed or missing final answers count incorrect.

| Condition | Correct / 392 | Accuracy | Parse failures / 392 | Correct in all four languages |
|---|---:|---:|---:|---:|
| Untouched base | 273 | 69.6% | 48 | 39 / 98 |
| English-only LoRA | 195 | 49.7% | 26 | 17 / 98 |
| Multilingual LoRA | 208 | 53.1% | 19 | 23 / 98 |

| Question language | Base | English-only LoRA | Multilingual LoRA |
|---|---:|---:|---:|
| English | 73 / 98 (74.5%) | 50 / 98 (51.0%) | 58 / 98 (59.2%) |
| Chinese | 79 / 98 (80.6%) | 60 / 98 (61.2%) | 55 / 98 (56.1%) |
| German | 65 / 98 (66.3%) | 47 / 98 (48.0%) | 49 / 98 (50.0%) |
| Japanese | 56 / 98 (57.1%) | 38 / 98 (38.8%) | 46 / 98 (46.9%) |

The primary paired contrast, **multilingual minus English-only LoRA**, is **+3.3 percentage points** (canonical-question-clustered bootstrap 95% CI **−2.3 to +8.9 points**). Per-language exact two-sided McNemar tests for that contrast have unadjusted p-values from **0.169 to 0.839**. These results do not establish a benefit from multilingual fine-tuning. Against the base, English-only LoRA is −19.9 points (95% CI −25.5 to −14.3) and multilingual LoRA is −16.6 points (−23.2 to −9.9). Lower parse-failure rates after fine-tuning did **not** translate into higher answer accuracy.

The [final research report](docs/research_report.md) explains language gaps, uncertainty, limitations, and the distinction between observed effects and possible causes. [Published aggregate evidence](docs/evidence/phase16l/README.md) contains the unmodified summary tables/JSON and artifact-hash manifest. The raw 1,176-response JSONL and translated questions are intentionally excluded from Git; the original run is archived separately in Google Drive at `multilingual-llm-reasoning/phase16l-final-evaluation-v1/final/`.

Private research files also have a **local sibling archive**, `../Multilingual LLM Research Archive/`, outside this Git checkout. It contains verified copies of historical outputs, intermediate datasets, the downloaded combined final predictions, and both complete trained LoRA adapter folders, including their original evaluation provenance. Required frozen inputs and baseline files may remain under ignored `outputs/` and `data/` paths so the existing scripts can still read them. The [archive and restore guide](docs/research_archive.md) explains the layout, checksums, and safe restoration. Adapters are never part of this Git repository.

## Engineering and methodology

- **Stack:** Python 3.11+, PyTorch, Hugging Face Datasets and Transformers, PEFT/TRL/bitsandbytes QLoRA, Pandas, Matplotlib, pytest, Ruff, YAML configs, JSONL/CSV/SQLite, and Colab CUDA. The base evaluation ran on Apple Silicon MPS; adapter evaluation ran on CUDA.
- **Data integrity:** stable canonical IDs trace GSM8K official train/test origins. The two frozen SFT conditions used the **same 197 training IDs** with exactly one question variant per ID; evaluation uses disjoint official-test IDs. English-only used 197 English questions. Multilingual used 50 English, 50 Chinese, 48 German, and 49 Japanese questions. Both retained the original **English worked solutions** and gold answers.
- **Training controls:** the two adapters used the same pinned base revision, QLoRA architecture, optimizer settings, effective batch size 8, seed 42, and **75 optimizer steps**. Assistant response tokens alone contributed supervised loss. The adapters were trained independently; no base weights are stored in this repository.
- **Inference controls:** the fixed Qwen chat template, an English `FINAL_ANSWER` instruction, greedy decoding, FP16, 256 maximum new tokens, and parser `final-answer-v2` were shared. Saved base predictions were reused only after an exact input/protocol/hash audit. Each adapter condition had its own resumable SQLite progress store and fresh model process.
- **Analysis controls:** strict ID/language/condition joins reject missing or duplicate predictions. The comparison uses 5,000 paired bootstrap resamples clustered by canonical question ID (seed 42), plus exploratory exact McNemar tests per language. Raw responses and parse failures remain traceable.

The fixed model revision is `aa8e72537993ba99e69dfaafa59ed015b17504d1`; GSM8K `main` was loaded at `740312add88f781978c0658806c59bc2815b9866`. The [Phase 16L run guide](docs/evaluation/phase16l_colab_evaluation.md) and [Colab notebook](notebooks/phase16l_colab_final_evaluation.ipynb) document the completed workflow. They retain explicit execution switches so opening the notebook never starts GPU evaluation.

## Install and verify without a GPU

From the repository root, with Python 3.11+:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev,analysis]'
python -m multilingual_llm_reasoning --help
python -m multilingual_llm_reasoning validate --config configs/dry_run.yaml
python -m multilingual_llm_reasoning run --config configs/dry_run.yaml --dry-run
pytest
ruff check .
ruff format --check .
pytest tests/test_published_evidence.py
```

The dry run uses a tiny synthetic fixture and needs no GPU, API key, GSM8K download, or model weights. Tests do not start training or inference. Five saved-run integrity tests use private Phase 16L inputs when they are restored locally and otherwise skip explicitly; all other tests use published evidence or fixtures/fake providers. Plotting tests use the `analysis` extra. Output directories are protected against accidental overwrite.

Reproducing the **real** study requires the separately archived frozen SFT datasets, translation drafts/review records, both Phase 16K LoRA adapters, and the original 98-question test grid. These are intentionally Git-ignored. Use the saved hashes in [the Phase 16L manifest](docs/evidence/phase16l/reproducibility_manifest.json), the frozen-data manifests, the versioned configs in `configs/`, and the [Colab guide](docs/evaluation/phase16l_colab_evaluation.md) to verify those inputs before any rerun. A fresh GitHub clone reproduces the software checks and published aggregate analysis, **not** the GPU experiment by itself.

The final workflow is: prepare and audit GSM8K/translation inputs with the retained data-preparation and QA modules; freeze the matched SFT files with `scripts/phase16h_freeze.py`; verify the frozen hashes and CUDA environment with `scripts/phase16i_cuda_preflight.py`; use the [smoke-test notebook](notebooks/phase16j_colab_training.ipynb) and [full-training guide](docs/training/phase16k_full_training_guide.md) for the two independent adapters; then use the [final-evaluation guide](docs/evaluation/phase16l_colab_evaluation.md). Numbered file names remain where they identify a hashed configuration or necessary provenance step. Those commands require the archived inputs and may consume GPU resources; the installation and evidence checks above do not.

## Repository layout

| Path | Purpose |
|---|---|
| `src/multilingual_llm_reasoning/` | Dataset, translation QA, inference, scoring, SFT, training, paired analysis, plotting |
| `configs/` | Versioned data, model, training, and evaluation settings |
| `scripts/` | Guarded data import, CUDA/Colab, persistence, and validation helpers |
| `notebooks/` | Opt-in Colab training and evaluation workflows |
| `tests/` | Offline regression tests for data integrity, parsing, caching, adapters, and statistics |
| `docs/research_report.md` | Final findings and limitations |
| `docs/evidence/phase16l/` | Publishable aggregate results and hashes; no raw questions or weights |
| `data/fixtures/` | Synthetic dry-run inputs only |
| `requirements/` | Pinned Colab dependency lock |

Generated `outputs/`, private `data/`, model caches, checkpoints, and the sibling Research Archive must never be staged. A fresh clone contains the synthetic fixture and publishable aggregate evidence only; it needs verified private inputs restored to the paths recorded in the final manifests before any full reproduction attempt.

## Limitations and responsible release

The 98-question translated evaluation set was **provisionally validated** (`research_ready: false`); full independent bilingual review of the non-English questions was not completed. The common instruction and all SFT solution targets are in English. Baseline inference used MPS FP16 while adapter inference used CUDA FP16. Only one base checkpoint, math dataset, training seed, and adapter budget were studied. Correct final answers do not establish faithful reasoning, and parse failures measure output compliance as well as task performance. Bootstrap intervals condition on the fixed translations, checkpoints, and seed; McNemar p-values are unadjusted for multiple comparisons. Plausible causes of the accuracy decline include adaptation budget, data scale, language/target mismatch, and translation noise, but this study does **not** isolate any cause.

The project code is released under the [MIT License](LICENSE), copyright © 2026 xhwa25. GSM8K's separate [upstream MIT notice](docs/licenses/GSM8K_LICENSE.txt) is retained. Raw GSM8K text, translations, model weights, adapters, caches, and credentials are excluded by `.gitignore`; the published aggregate tables contain no question text. Verify model and dataset terms before redistributing external assets. Before publishing, review `git status --short` and staged files; never force-add ignored datasets, weights, or secrets.
