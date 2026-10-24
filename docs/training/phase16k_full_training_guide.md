# Phase 16K: independent full QLoRA runs on Colab

The two full runs completed for the published study. This guide preserves their execution procedure; see the [final research report](../research_report.md) for results. The commands below start GPU work only when deliberately invoked in a suitable Colab environment.

This guide accompanies [the Phase 16K notebook](../../notebooks/phase16k_colab_full_training.ipynb). It prepares two independent full runs and does not itself start training. The notebook's `RUN_ENGLISH_FULL`, `RUN_MULTILINGUAL_FULL`, and `PERSIST_FULL_TO_DRIVE` switches all default to `False`. Enable each deliberately after the CUDA preflight passes. A T4 16 GB or better is preferred; actual free VRAM and NF4 compatibility must pass preflight.

## Transfer the exact project and frozen inputs

From the project root on the Mac, build a **new** allowlisted archive after obtaining this Phase 16K code:

```sh
python3 scripts/bundle_colab.py --phase16i-frozen --output outputs/phase16k-frozen-upload-v1.tar.gz
```

The archive includes the Phase 16H frozen English and multilingual datasets, the designated validation JSONL, the Phase 16I training manifest and four configurations, the existing trainer, and `scripts/phase16k_full_runner.py`. It excludes model weights, caches, credentials, and held-out evaluation outputs. The bundler refuses to overwrite an existing archive. Upload [the notebook](../../notebooks/phase16k_colab_full_training.ipynb) in Colab, choose **Runtime → Change runtime type → GPU**, and run setup cells in order. Upload the new archive when prompted; the notebook refuses to extract over an existing project directory.

The isolated `scripts/setup_colab.sh` installs the hash-locked Python 3.11.14 CUDA environment in `.venv-colab`. Its lock pins PyTorch `2.8.0+cu126`, Transformers `4.56.2`, PEFT `0.17.1`, TRL `0.23.1`, Accelerate `1.10.1`, datasets `4.1.1`, and bitsandbytes `0.47.0`. The notebook verifies the pinned Qwen revision `aa8e72537993ba99e69dfaafa59ed015b17504d1` and both Phase 16H SHA-256 hashes before full training. The preflight checks CUDA, free VRAM, package/API versions, bitsandbytes NF4, disk headroom, data integrity, and matched configs. Stop if it reports `blocked`.

## Fairness contract

The existing full configs are `configs/phase16i_english_lora.yaml` and `configs/phase16i_multilingual_lora.yaml`. `scripts/phase16k_full_runner.py` compares every effective config field except name, training condition, and output directory. The configs use the same pinned base/tokenizer revision, FP16 NF4 QLoRA, LoRA rank 16/alpha 32/dropout 0.05 with the same seven projection modules, `adamw_torch`, cosine schedule, learning rate `2e-4`, batch size 1 with gradient accumulation 8 (effective batch 8), maximum length 1024, seed 42, three epochs and exactly **75 optimizer steps**. The trainer preserves the audited response-only labels and ignores prompt/padding tokens. It uses only the 100-example designated validation set for loss diagnostics.

Each condition has 197 unique matched canonical training IDs. English-only uses 197 English questions. Multilingual uses 50 English, 50 Chinese, 48 German, and 49 Japanese questions. Both use the original English worked solutions. The English set has 23,113 prompt and 25,870 supervised response tokens (48,983 total); multilingual has 24,669 prompt and 25,870 response tokens (50,539 total). The 1,556-token, 3.18% prompt difference is reported without adjusting the optimizer-step or example budget. There is no held-out 98-question evaluation in this notebook.

The wrapper will not use a smoke checkpoint or the other condition's adapter. Fresh runs require empty, distinct local and Drive condition directories. It launches the existing trainer in a child process, so the English model and optimizer exit before multilingual training starts. The trainer initializes a fresh adapter on the same original pinned base for each run. Training outputs are under `outputs/training/phase16i_english_lora/full/` and `outputs/training/phase16i_multilingual_lora/full/` in the Colab project.

## Drive persistence during training

The notebook requires you to mount Drive and verify both Phase 16J smoke `run.json` files show `status=completed` and `global_step=20`. Set `SMOKE_DRIVE_ROOT` to the actual smoke-artifact location if it differs from the notebook default. Set `PERSIST_FULL_TO_DRIVE=True` before enabling either full-run switch. The proposed new destination is `MyDrive/multilingual-llm-reasoning/phase16k-full-training-v1/`; choose a new version if it is already occupied. The wrapper refuses an unmounted Drive, a foreign experiment contract, or an existing condition output.

The trainer saves checkpoints at optimizer steps **25, 50, and 75**, retaining up to two local checkpoints. The wrapper polls every five seconds and mirrors each complete checkpoint to the versioned Drive condition folder during training. It requires adapter weights, adapter config, trainer state, optimizer, scheduler, RNG, and FP16 scaler files; checks the step and 75-step budget; deserializes adapter, optimizer, scheduler, and scaler state; hashes all required files, including RNG state; copies to a unique staging folder; verifies source and destination hashes; then publishes the folder. It also snapshots `run.json` at that step. If mirroring fails, the wrapper terminates the trainer rather than silently continuing without persistence. A Colab disconnect can still occur before the first step-25 checkpoint; there is then no resumable state.

After 75 steps, the wrapper verifies `run.json`, finite final training/validation losses, checkpoint-75, frozen hashes, and the saved adapter. It loads that adapter onto a fresh pinned 4-bit base, checks explicit adapter activation, and executes a tiny forward pass. It then copies `adapter/`, logs, metrics, training state and verification report to Drive, checks their hashes, and writes `completion.json`. These checks have been prepared and unit tested locally; **no Phase 16K CUDA full run or resume has been executed by this preparation task**.

The wrapper saves and verifies `environment.freeze.txt` when the first run starts, so it survives an interruption. The notebook's final cell verifies both Drive results and that environment record, then writes `phase16k_training_completion_report.md` only after both runs actually complete. It refuses to overwrite an existing report. Preserve the entire Drive experiment folder, including checkpoint directories, if resume or audit is needed.

## Exact Colab commands behind the switches

From `/content/multilingual-llm-reasoning`, after mounting Drive and passing preflight:

```sh
.venv-colab/bin/python scripts/phase16i_cuda_preflight.py --output outputs/phase16k-colab/cuda_preflight.json
CUDA_VISIBLE_DEVICES=0 .venv-colab/bin/python scripts/phase16k_full_runner.py --condition english --drive-root /content/drive/MyDrive/multilingual-llm-reasoning/phase16k-full-training-v1 --confirm-full-training
CUDA_VISIBLE_DEVICES=0 .venv-colab/bin/python scripts/phase16k_full_runner.py --condition multilingual --drive-root /content/drive/MyDrive/multilingual-llm-reasoning/phase16k-full-training-v1 --confirm-full-training
```

Use notebook switches to invoke these commands one at a time. Do not rerun a successful full-run cell: it will refuse existing outputs. If interrupted, follow [the checkpoint recovery guide](phase16k_checkpoint_recovery.md). Use [the completion checklist](phase16k_completion_checklist.md) before treating the adapters as frozen. Final held-out model comparison is a separate phase.
