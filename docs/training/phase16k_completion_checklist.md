# Phase 16K full-training completion checklist

All execution items remain **pending** until the two real Colab CUDA runs finish. The reported Phase 16J smoke losses and memory usage are prerequisites, not Phase 16K outcomes.

| Check | Required evidence | Initial status |
|---|---|---|
| Fresh CUDA preflight | `outputs/phase16k-colab/cuda_preflight.json`: ready, NF4 passed, no blockers | Pending |
| Frozen data unchanged | Both Phase 16H SHA-256 hashes match before and after runs | Pending |
| Matched configurations | `experiment_contract.json` and local paired-config validation | Pending |
| English independent run | English `run.json`: completed, full mode, no smoke resume | Pending |
| Multilingual independent run | Multilingual `run.json`: completed, full mode, no English/smoke resume | Pending |
| Exact optimizer budget | Both `metrics.json`: `global_step=75`; checkpoint-75 trainer state agrees | Pending |
| Finite supervised losses | Both final train and designated-validation losses are finite | Pending |
| Response-only loss | Existing audited `labels` and collator tests pass; no prompt or padding supervision | Pending |
| In-flight Drive backups | Verified checkpoint integrity manifests at intermediate step 25 or 50 and final step 75 for each condition | Pending |
| Adapter loads and activates | Both `adapter_load_verification.json`: passed on fresh pinned 4-bit base | Pending |
| Saved final artifacts | Both Drive `final/` folders contain adapter, metrics, logs, provenance, and completion JSON | Pending |
| Resource reporting | Runtime, peak allocated/reserved GPU memory, learning rate, seed, batch, 197 examples and token counts recorded | Pending |
| Fairness review | Same 75-step/effective-batch/example budget; 3.18% prompt-token difference disclosed | Pending |
| No held-out test use | No evaluation on the 98-question multilingual test set during training | Pending |
| Final report | Versioned `phase16k_training_completion_report.md` written only after both runs pass | Pending |

The notebook will display pending status where outputs are absent. Do not mark the adapters ready for final evaluation from a saved checkpoint alone. The training guide explains the run procedure; the recovery guide describes interrupted-run handling and its validation limit.
