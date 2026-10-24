# Phase 16L paired comparisons — provisional

Accuracy and primary parse-failure rate divide by all expected originals per language. Inference failures/pending count incorrect and are reported separately. No rows dropped.

paired percentile, linear quantiles, shared resamples; 5000 resamples; unit: original problem ID.
Exact two-sided McNemar tests are per language, paired by canonical ID. No pooled 392-row independence assumption or multiple-testing correction.
Exploratory unadjusted 95% intervals; no multiple-comparison correction. With 100 originals, each language accuracy changes by 1 percentage point per question. Bootstrap intervals can be degenerate and omit translation/model/seed uncertainty; not proof of improvement.

| Treatment | Reference | Metric | Difference | 95% CI |
|---|---|---|---:|---:|
| english_lora | base | en | -0.235 | [-0.347, -0.122] |
| english_lora | base | zh | -0.194 | [-0.296, -0.102] |
| english_lora | base | de | -0.184 | [-0.286, -0.082] |
| english_lora | base | ja | -0.184 | [-0.286, -0.082] |
| english_lora | base | macro_accuracy | -0.199 | [-0.255, -0.143] |
| english_lora | base | all_languages_correct | -0.224 | [-0.327, -0.122] |
| multilingual_lora | base | en | -0.153 | [-0.255, -0.051] |
| multilingual_lora | base | zh | -0.245 | [-0.357, -0.133] |
| multilingual_lora | base | de | -0.163 | [-0.265, -0.071] |
| multilingual_lora | base | ja | -0.102 | [-0.204, 0.010] |
| multilingual_lora | base | macro_accuracy | -0.166 | [-0.232, -0.099] |
| multilingual_lora | base | all_languages_correct | -0.163 | [-0.265, -0.061] |
| multilingual_lora | english_lora | en | 0.082 | [-0.020, 0.184] |
| multilingual_lora | english_lora | zh | -0.051 | [-0.163, 0.061] |
| multilingual_lora | english_lora | de | 0.020 | [-0.082, 0.122] |
| multilingual_lora | english_lora | ja | 0.082 | [-0.031, 0.194] |
| multilingual_lora | english_lora | macro_accuracy | 0.033 | [-0.023, 0.089] |
| multilingual_lora | english_lora | all_languages_correct | 0.061 | [-0.020, 0.143] |

| Treatment | Reference | Language | Discordant | Exact two-sided p |
|---|---|---|---:|---:|
| english_lora | base | en | 39 | 0.0003 |
| english_lora | base | zh | 29 | 0.0005 |
| english_lora | base | de | 32 | 0.0021 |
| english_lora | base | ja | 32 | 0.0021 |
| multilingual_lora | base | en | 29 | 0.0081 |
| multilingual_lora | base | zh | 36 | 0.0001 |
| multilingual_lora | base | de | 26 | 0.0025 |
| multilingual_lora | base | ja | 30 | 0.0987 |
| multilingual_lora | english_lora | en | 26 | 0.1686 |
| multilingual_lora | english_lora | zh | 31 | 0.4731 |
| multilingual_lora | english_lora | de | 24 | 0.8388 |
| multilingual_lora | english_lora | ja | 30 | 0.2005 |
