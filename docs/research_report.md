# Do LLMs Reason Equally Well Across Languages?

*A paired evaluation of Qwen2.5-3B-Instruct and two LoRA adapters on translated GSM8K mathematical word problems · final project report · October 2026*

## Abstract

This project tested whether prompt language changes numerical answer accuracy for equivalent grade-school math problems, and whether multilingual-question fine-tuning changes that pattern. Three conditions—the untouched `Qwen/Qwen2.5-3B-Instruct` model, an English-only LoRA adapter, and a multilingual-question LoRA adapter—answered the same 98 canonical GSM8K test questions in English, Chinese, German, and Japanese. All 1,176 expected responses completed. Overall accuracy was 69.6% for the base, 49.7% for English-only LoRA, and 53.1% for multilingual LoRA. Both adapters underperformed the base. The multilingual adapter exceeded English-only LoRA by 3.3 percentage points, but its paired canonical-question bootstrap 95% interval was −2.3 to +8.9 points. The translated evaluation set received limited QA and remains provisionally validated, so these outcomes are exploratory. The project also delivers reproducible dataset, translation-review, QLoRA training, caching, scoring, and paired-analysis infrastructure.

## Questions, hypotheses, and scope

The primary question was whether changing only a mathematical word problem's language affects exact numerical answer accuracy. The adapter comparison asked whether multilingual-question exposure, under a matched training budget and original problem IDs, improves outcomes relative to English-only exposure. Proposed directional expectations were that English might outperform other languages and that multilingual training might narrow gaps. The observed base model instead scored highest in Chinese, and neither adapter improved overall accuracy over the base. This study concerns translated GSM8K math questions only; it does not measure formal logic, general multilingual competence, or human ability.

This portfolio report uses measured project artifacts and does not make literature-priority claims. A formal related-work section with verified primary citations would be required for academic submission; unverified references are intentionally omitted.

## Data and translation QA

The source is Hugging Face `openai/gsm8k`, configuration `main`, pinned dataset revision `740312add88f781978c0658806c59bc2815b9866`. The official train/test boundary was preserved. Stable canonical hashes identify original questions across all variants, training conditions, and results. The held-out evaluation grid contains 98 original official-test questions, each represented once in `en`, `zh`, `de`, and `ja` (392 variants); the two previously identified ambiguous questions were excluded before the final grid was frozen. Gold numerical answers came from GSM8K's `####` marker. The evaluation selection and exclusion were not based on the final three-condition outcomes.

Translation work compared local NLLB and Qwen drafts, applied deterministic checks for quantities, units, final questions, rounding, relations, and answer leakage, and prepared review artifacts. Those checks can flag risk but cannot prove semantic equivalence. The 98-question final set was permitted for a **limited-QA, provisional evaluation**, with some translation risks unresolved and human approval statuses pending. Its manifest records `research_ready: false`. Consequently, language differences cannot be attributed solely to the model: source wording, translation choices, and undetected semantic changes may contribute. Raw translated questions are not published in this repository.

The approved SFT freeze used 197 original GSM8K training IDs in both conditions, exactly one example per ID and no overlap with the held-out evaluation IDs. English-only training used 197 English questions. Multilingual-question training assigned 50 English, 50 Chinese, 48 German, and 49 Japanese questions. The **worked solutions remained in English in both conditions**; this is question-language exposure, not multilingual solution supervision. Three disputed translated training questions were excluded symmetrically. The 147 retained non-English translations were approved by explicit project-owner authorization following AI-assisted semantic review, not independent expert verification. The frozen training files and their SHA-256 hashes are recorded in the Phase 16H/16I manifests.

## Controlled training and inference

All conditions use `Qwen/Qwen2.5-3B-Instruct` revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`. The two QLoRA adapters used matched rank 16, alpha 32, dropout 0.05, target projection modules, optimizer schedule, effective batch size 8, seed 42, and 75 optimizer steps. Training used assistant-response-only loss masking. Equal original IDs and steps do not equalize input token counts, GPU time, or training randomness. Only one training seed and budget were evaluated; the project does not establish general behavior across fine-tuning regimes.

The inference protocol applied the same chat template, English instruction wrapper, greedy decoding, FP16, 256-new-token cap, and `final-answer-v2` parser to all languages and conditions. The English system instruction is consistent but not linguistically neutral. The original base responses were reused after exact-input, prompt, decoding, checkpoint, parser, and hash audits; this avoided unnecessary repeated inference. The base run used Apple Silicon MPS and adapter runs used NVIDIA CUDA, both in FP16. Cross-device numerical behavior can differ despite the matched protocol. Each adapter was loaded independently on a fresh base process, with active-adapter checks and a resumable SQLite progress store. The final comparison joined all 1,176 expected condition/ID/language keys; no completed variant was dropped from the denominator.

## Results

All numbers below are from the saved [Phase 16L summary](evidence/phase16l/summary.json), [accuracy tables](evidence/phase16l/summary.csv), and [statistical comparison](evidence/phase16l/statistical_comparison.md). The downloaded combined predictions contain 1,176 unique completed rows; the published aggregate evidence excludes question text and raw responses.

| Condition | Correct / 392 | Accuracy | Parse failures | Parsed-output accuracy* | All four languages correct |
|---|---:|---:|---:|---:|---:|
| Base | 273 | 69.6% | 48 | 273 / 344 = 79.4% | 39 / 98 |
| English-only LoRA | 195 | 49.7% | 26 | 195 / 366 = 53.3% | 17 / 98 |
| Multilingual LoRA | 208 | 53.1% | 19 | 208 / 373 = 55.8% | 23 / 98 |

*Parsed-output accuracy is descriptive, conditional on a successful parse; the primary accuracy denominator is always all 392 expected responses. Parse failures are counted incorrect, never silently removed.*

| Language (98 questions each) | Base | English-only LoRA | Multilingual LoRA |
|---|---:|---:|---:|
| English | 73 (74.5%) | 50 (51.0%) | 58 (59.2%) |
| Chinese | 79 (80.6%) | 60 (61.2%) | 55 (56.1%) |
| German | 65 (66.3%) | 47 (48.0%) | 49 (50.0%) |
| Japanese | 56 (57.1%) | 38 (38.8%) | 46 (46.9%) |

The base model's Chinese accuracy exceeded its English accuracy by 6.1 points; English exceeded German by 8.2 points and Japanese by 17.3 points. Those are paired within-question gaps, not evidence that Chinese questions are intrinsically easier or that translation quality is equal. After English-only LoRA, Chinese remained 10.2 points above English, whereas Japanese was 12.2 points below. The multilingual adapter put English 3.1 points above Chinese, 9.2 above German, and 12.2 above Japanese. The [saved gap intervals](evidence/phase16l/english_gaps.csv) show which of these exploratory gaps include zero. Correct-in-all-four-language consistency was 39.8% for base, 17.3% for English-only LoRA, and 23.5% for multilingual LoRA.

A parse failure means the output did not satisfy the strict terminal `FINAL_ANSWER: <number>` policy; it is not synonymous with a wrong mathematical computation. The base had 44 missing-marker and four invalid-number failures. English-only LoRA had 23 and three, respectively; multilingual LoRA had 18 and one. There were no inference failures or pending rows. Although both adapters reduced parse failures, they had substantially lower numerical accuracy. A parser-compliant output can still contain a wrong answer, and a failed parse can sometimes contain reasoning text whose correctness is not established by this scoring protocol. The original responses were not rewritten or reparsed with a new policy for this report.

## Paired uncertainty and interpretation

Differences use the same canonical questions under all conditions. The 95% percentile intervals came from 5,000 shared bootstrap resamples of **original problem IDs**, so the four language variants of a question stay together. The primary multilingual-minus-English-only macro accuracy difference is **+3.3 points**, 95% CI **−2.3 to +8.9**. The corresponding language-specific differences are +8.2 in English (CI −2.0 to +18.4), −5.1 in Chinese (−16.3 to +6.1), +2.0 in German (−8.2 to +12.2), and +8.2 in Japanese (−3.1 to +19.4). Every interval includes zero. Exact two-sided per-language McNemar p-values for this adapter contrast are 0.169, 0.473, 0.839, and 0.200, respectively. They are unadjusted exploratory tests, and no pooled test treats all 392 variants as independent.

Relative to base, the English-only adapter's macro accuracy declined **19.9 points** (95% CI −25.5 to −14.3); multilingual LoRA declined **16.6 points** (−23.2 to −9.9). The weaker adapter scores are therefore clear for this fixed design and sample, while the small difference between adapters remains uncertain. Lower parsing failures show improved output-format compliance, not improved reasoning. Possible contributors to accuracy degradation include a small 197-question training set, a 75-step budget, English-only solution targets, translated-question noise, or sensitivity to fine-tuning settings. None was isolated experimentally, so the report does not assign a causal explanation.

Bootstrap intervals quantify variation over these 98 sampled original questions, conditional on fixed translations, checkpoints, prompt, parser, and training seed. They do not cover translation error, selection effects, repeated tuning, device differences, model stochasticity across training seeds, or performance on other math distributions. The many language/condition tests were not multiplicity-corrected. One changed answer moves a per-language accuracy by about 1.02 points. These constraints limit generalizability and make a claim of multilingual fine-tuning benefit unsupported.

## Qualitative analysis boundary

The code can generate question-linked disagreement reports and review CSVs without assigning subjective error categories as ground truth. This final documentation does **not** claim a completed independent bilingual error audit or attribute particular wrong answers to arithmetic, comprehension, or translation. Numeric and final-question QA flags are heuristics. Correct answers do not prove faithful stepwise reasoning, and raw output inspection alone may not reveal why a model erred.

## Reproducibility, file hygiene, and ethics

The [README](../README.md) gives a GPU-free installation/test command and explains the separate Colab training/evaluation workflows. YAML configs, pinned revisions, seed, data hashes, frozen SFT manifests, adapter provenance, and SQLite cache signatures document the experiment. The [Phase 16L reproducibility manifest](evidence/phase16l/reproducibility_manifest.json) records hashes for the test set, saved baseline, adapters, configs, and final artifacts. The 13 files supplied in the downloaded final folder matched their recorded hashes. The manifest also lists three per-condition prediction files absent from that download, so their individual hashes were not independently checked here; the combined 1,176-row file and summary were checked. Published aggregate evidence is a subset of the archived run, not a complete raw replication package.

The project code has a separate [MIT License](../LICENSE), copyright © 2026 xhwa25; GSM8K's [upstream MIT notice](licenses/GSM8K_LICENSE.txt) accompanies the project. Raw questions and translations, LoRA weights, model caches, checkpoints, credentials, and private review files are excluded from Git. Dataset and model licenses remain distinct from the project code license. The translation-review status, English wrapper and targets, single model/seed/budget, and provisional test set must remain visible whenever these results are presented. Cross-language scores concern this dataset and protocol, not the abilities of people who use those languages.
