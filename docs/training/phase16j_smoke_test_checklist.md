# Phase 16J CUDA smoke-test checklist

This is an evidence checklist for the two 20-optimizer-step QLoRA runs. All GPU execution items are **pending** until a Colab CUDA session actually produces the named artifacts. Do not infer success from a dry run or mark them complete on the Mac.

| Check | Evidence to inspect | Status before Colab |
|---|---|---|
| NVIDIA GPU and driver visible | `nvidia-smi` cell; GPU name and free VRAM | Pending |
| Locked isolated environment installed | `.venv-colab/environment.freeze.txt`; setup exit code 0 | Pending |
| Pinned Qwen revision accessible | Notebook Hub metadata SHA matches `aa8e72537993ba99e69dfaafa59ed015b17504d1` | Pending |
| Frozen data unchanged | Notebook SHA-256 checks and `cuda_preflight.json` hashes | Pending |
| CUDA/QLoRA preflight passes | `outputs/phase16j-colab/cuda_preflight.json`: `status=ready`, `qlora_kernel_probe=passed`, no blockers | Pending |
| English 20-step smoke passes | `outputs/training/phase16i_english_smoke/smoke/run.json`: `status=completed`; `metrics.json`: `global_step=20` | Pending |
| Multilingual 20-step smoke passes | `outputs/training/phase16i_multilingual_smoke/smoke/run.json`: `status=completed`; `metrics.json`: `global_step=20` | Pending |
| Finite losses | Every loss field in each `metrics.json` and `training_log.jsonl` is finite | Pending |
| Backward/optimizer update | `smoke_optimizer_update_verified=true`; 20 optimizer steps, not 20 microbatches | Pending |
| Checkpoint saved and reloaded | `smoke_checkpoint_load_verified=true`; `checkpoint-20/` contains adapter, optimizer, scheduler, trainer and RNG state | Pending |
| No CUDA OOM or CPU/MPS fallback | Training exits 0; `run.json` has no error; preflight and trainer report CUDA device | Pending |
| Peak GPU memory and duration recorded | Each `metrics.json`: `peak_allocated_gib`, `peak_reserved_gib`, `runtime_seconds` | Pending |
| Frozen datasets still unchanged | Recompute both SHA-256 hashes after the runs; compare with Phase 16H | Pending |
| Outputs preserved | Optional Drive copy of both complete smoke folders, preflight, configs and environment freeze | Pending |

The trainer already validates that only LoRA parameters are trainable, the loaded Qwen tokenizer matches the frozen token IDs and chat template, prompt tokens have label `-100`, padding labels are ignored, training/evaluation losses are finite, adapter weights change, and the step-20 checkpoint can be loaded. The notebook result cell checks those recorded outcomes. If any item fails, keep the failure artifact and report the exact error. Do not turn a partial run into a passing result or proceed to full training.

Before comparing the two conditions, confirm both YAML files still match in model/revision, QLoRA and precision, LoRA modules/rank/alpha/dropout, optimizer, learning rate/scheduler, effective batch size, sequence length, seed and 20-step budget. The Phase 16I preflight performs this comparison. The training datasets are the intended difference; the 197 canonical IDs are matched across conditions. The multilingual dataset has slightly more total prompt tokens, so matching example counts and optimizer steps does not imply equal token exposure.

Exact Colab commands and Drive instructions are in [the setup guide](phase16j_colab_setup.md). This checklist is intentionally unfilled until a real CUDA run occurs.
