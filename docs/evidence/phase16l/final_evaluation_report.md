# Phase 16L held-out multilingual evaluation — provisional

The three conditions share 98 canonical GSM8K test questions and four language variants per question (1,176 responses). The translation dataset has limited QA, pending human approval, and research_ready: false.

The base predictions were generated on MPS FP16; the LoRA predictions use CUDA FP16. Model revision, prompt, decoding, parser, and exact test inputs were audited for reuse. Cross-device numerical differences remain possible.

| Condition | Correct / 392 | Accuracy | Parse failures / 392 |
|---|---:|---:|---:|
| base | 273/392 | 0.696 | 48/392 |
| english_lora | 195/392 | 0.497 | 26/392 |
| multilingual_lora | 208/392 | 0.531 | 19/392 |

Per-language denominators, paired uncertainty, and exact McNemar results are in the saved CSV/JSON files and statistical_comparison.md. Parse failures count incorrect. The instruction wrapper is English across all languages. These exploratory results do not establish a fine-tuning benefit without interpreting uncertainty and translation risks.
