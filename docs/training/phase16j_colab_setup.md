# Phase 16J: Google Colab CUDA setup

Use [the Phase 16J notebook](../../notebooks/phase16j_colab_training.ipynb) on a Google Colab NVIDIA GPU runtime. A T4 16 GB or better is preferred, but the existing preflight must pass before training. The notebook does not request a paid runtime, and both 20-step smoke switches default to `False`. It never runs full training.

## Build and upload the frozen project

From the project root on the Mac, create a new allowlisted archive:

```sh
python3 scripts/bundle_colab.py --phase16i-frozen --output outputs/phase16j-frozen-upload.tar.gz
```

The flag adds the approved Phase 16H freeze manifest and two SFT JSONL files, the Phase 16C validation JSONL, and the Phase 16I training manifest. It does not include credentials, model weights, caches, evaluation results, or unrelated output files. The source files are copied byte for byte. If the archive already exists, choose a new name; the bundler refuses to overwrite it.

Upload the notebook in Colab, select **Runtime → Change runtime type → GPU**, and execute cells in order. The repository setup cell asks for the archive. An alternative is to clone a pinned Git commit into `/content/multilingual-llm-reasoning`, then separately place those five frozen files at their original relative paths. Do not expect a Git clone alone to include untracked frozen datasets. The archive route is the tested project workflow.

The dataset SHA-256 values checked in the notebook and preflight are:

| Frozen file | SHA-256 |
|---|---|
| `english_only_sft_final.jsonl` | `6e59b5344db55eb84f87c9dc9059587fbd00dae0edcea967a76d37705b8ed881` |
| `multilingual_sft_final.jsonl` | `9b5c813da240229a31a7e57d1bc8a8e4cb9e8a410a837c38b01db23202df2021` |

## Install and preflight

The notebook calls `bash scripts/setup_colab.sh`. That script creates an isolated Python 3.11.14 environment in `.venv-colab`, installs the hash-locked `requirements/colab.lock`, checks dependency consistency, and records `.venv-colab/environment.freeze.txt`. It keeps Colab's notebook Python separate. The lock pins PyTorch `2.8.0+cu126`, Transformers `4.56.2`, PEFT `0.17.1`, TRL `0.23.1`, Accelerate `1.10.1`, datasets `4.1.1`, and bitsandbytes `0.47.0`. A fresh environment downloads CUDA packages. The pinned Qwen weights are only fetched when an explicitly enabled smoke run starts.

The notebook checks model Hub metadata for `Qwen/Qwen2.5-3B-Instruct` revision `aa8e72537993ba99e69dfaafa59ed015b17504d1` without downloading weights. Then it runs:

```sh
.venv-colab/bin/python scripts/phase16i_cuda_preflight.py --output outputs/phase16j-colab/cuda_preflight.json
```

The preflight compares the paired YAML settings, validates the frozen hashes and SFT split, probes package/API compatibility, runs a bitsandbytes NF4 CUDA kernel check, and reports GPU model, free VRAM, and free disk. It requires at least 10 GiB free VRAM for the current QLoRA profile and 20 GiB free disk. This is an admission estimate, not an out-of-memory guarantee. Stop if `status` is `blocked`. Do not silently fall back to CPU/MPS or alter only one condition's settings.

## Run the two smoke tests

After reviewing preflight, set each notebook switch to `True` and run its cell deliberately. The equivalent commands, from the project root, are:

```sh
CUDA_VISIBLE_DEVICES=0 .venv-colab/bin/python -m multilingual_llm_reasoning train --config configs/phase16i_english_smoke.yaml --mode smoke --confirm-gpu-training
CUDA_VISIBLE_DEVICES=0 .venv-colab/bin/python -m multilingual_llm_reasoning train --config configs/phase16i_multilingual_smoke.yaml --mode smoke --confirm-gpu-training
```

Both configurations use Qwen2.5-3B-Instruct at the same revision, NF4 QLoRA with FP16 compute, LoRA rank 16/alpha 32/dropout 0.05 and identical target modules, `adamw_torch`, cosine scheduling, learning rate `2e-4`, batch size 1 with gradient accumulation 8 (effective batch size 8), maximum sequence length 1024, and seed 42. `--mode smoke` overrides the duration to **exactly 20 optimizer steps** and saves/evaluates every 5 steps. The only intended difference is the approved training dataset. The English set has 197 English examples; the multilingual set has 197 matched IDs with 50 English, 50 Chinese, 48 German, and 49 Japanese questions. Both retain the original English worked solutions, a study limitation.

The trainer validates response-only labels, prompt and padding masking, tokenizer identity, frozen data, finite train/evaluation losses, forward/backward and optimizer progress, changed LoRA adapter weights, an adapter-only checkpoint at step 20, and deserializable adapter/optimizer/scheduler state. It writes `run.json`, `metrics.json`, `training_log.jsonl`, `checkpoint-20/`, and `adapter/` under distinct smoke directories. The notebook's comparison cell verifies completion and prints measured runtime and peak allocated/reserved GPU memory. A dry run or the presence of a directory alone is not a smoke-test pass.

If a run stops after a complete earlier checkpoint, inspect `run.json` and the checkpoint files before using the existing `--resume latest` option with the same config and mode. The trainer refuses a mismatched identity or completed budget. If CUDA OOM occurs, stop and record it; select a GPU with more free VRAM, or revise both configs and revalidate the experiment together. Do not claim matched conditions after an asymmetric change.

## Preserve artifacts

Colab storage is ephemeral. The notebook's Drive persistence cell is disabled by default and copies both completed smoke directories, the preflight report, environment freeze, and smoke configs to a new, non-overwriting folder in `MyDrive`. Enable it only after the two result checks pass. Keep the whole `checkpoint-20/` directory if exact resume is needed; an `adapter/` folder alone is for inference, not optimizer resume. Do not put Hugging Face tokens or other secrets in the notebook, archive, or Drive checkpoint folder. Use Colab's private secrets mechanism if access to the pinned public model changes.

## Eventual full training (not part of Phase 16J)

Only after both 20-step smoke runs pass and full training is separately authorized, verify the two full-run dry runs and launch independent sessions:

```sh
CUDA_VISIBLE_DEVICES=0 .venv-colab/bin/python -m multilingual_llm_reasoning train --config configs/phase16i_english_lora.yaml --mode dry-run --check-hardware
CUDA_VISIBLE_DEVICES=0 .venv-colab/bin/python -m multilingual_llm_reasoning train --config configs/phase16i_multilingual_lora.yaml --mode dry-run --check-hardware
# Later, after separate authorization:
CUDA_VISIBLE_DEVICES=0 .venv-colab/bin/python -m multilingual_llm_reasoning train --config configs/phase16i_english_lora.yaml --mode full --confirm-gpu-training
CUDA_VISIBLE_DEVICES=0 .venv-colab/bin/python -m multilingual_llm_reasoning train --config configs/phase16i_multilingual_lora.yaml --mode full --confirm-gpu-training
```

The full profile targets 75 optimizer steps per condition. Check the actual dry-run report before launching. Full-run output directories are separate from smoke outputs. For an interrupted full run, restore its complete checkpoint and `run.json` to the original relative output path, then use the same command with `--resume latest`. See the [full-training guide](phase16k_full_training_guide.md) for trainer details and the [smoke checklist](phase16j_smoke_test_checklist.md) for acceptance evidence.
