# Phase 16K checkpoint recovery

The existing trainer writes adapter-only checkpoint folders every 25 optimizer steps, with optimizer, scheduler, RNG, FP16 scaler, and `trainer_state.json`. `scripts/phase16k_full_runner.py` mirrors and hashes complete checkpoints on Drive while the run is active. Its restore path is unit tested with offline fixtures. **A real interrupted CUDA resume has not yet been demonstrated**, so inspect the resumed step and losses on the GPU before relying on this for an expensive run.

1. Start a fresh Colab NVIDIA GPU runtime. Upload the same pinned project/frozen-data bundle, install the locked environment, mount the same Drive folder, and rerun the CUDA preflight. Do not change a YAML config, dataset, model revision, or tokenizer revision.
2. Inspect `MyDrive/multilingual-llm-reasoning/phase16k-full-training-v1/<condition>/checkpoints/`. A folder is eligible only if it has `checkpoint_integrity.json`, all required state files, and an optimizer step below 75. The companion `run_at_checkpoint-<step>.json` must exist.
3. Ensure the new local `outputs/training/phase16i_<condition>_lora/full/` directory is empty. The wrapper refuses to restore over a nonempty local run.
4. Run the appropriate recovery command below. It verifies all checkpoint hashes and the saved full-run identity, restores the highest complete checkpoint below 75, and calls the existing trainer with `--mode full --resume latest --confirm-gpu-training`. The full config still caps the run at 75 total optimizer steps; it does not add 75 new steps.

```sh
CUDA_VISIBLE_DEVICES=0 .venv-colab/bin/python scripts/phase16k_full_runner.py --condition english --drive-root /content/drive/MyDrive/multilingual-llm-reasoning/phase16k-full-training-v1 --resume-from-drive --confirm-full-training
CUDA_VISIBLE_DEVICES=0 .venv-colab/bin/python scripts/phase16k_full_runner.py --condition multilingual --drive-root /content/drive/MyDrive/multilingual-llm-reasoning/phase16k-full-training-v1 --resume-from-drive --confirm-full-training
```

Run **only** the command for the interrupted condition. Do not use a Phase 16J smoke checkpoint, the other condition's checkpoint, or a checkpoint at step 75. The wrapper rejects a different experiment contract or completed final folder. If a training run finished all 75 steps but final adapter verification or Drive copy failed, `--resume-from-drive` cannot add training steps; diagnose and verify the existing local/Drive files instead of restarting.

After resume, inspect local and Drive `run.json`/`completion.json`: require `status=completed`, `global_step=75`, finite losses, a verified `checkpoint-75`, `adapter_load_verification.json` with `status=passed`, and unchanged frozen dataset hashes. A mismatched identity, corrupt/incomplete state, or failed checkpoint copy is a blocker. A `.partial` Drive folder is not a published checkpoint; preserve it for diagnosis, but never use it for resume. If Drive disconnects during training, the wrapper stops the child trainer rather than continuing without checkpoint persistence.

The trainer's existing resume contract checks adapter, optimizer, scheduler, RNG and FP16 scaler files plus matching configuration/data identity. This establishes that the required files and restore call are present; the first actual CUDA interruption/recovery should still be treated as a validation exercise, not an already-proven result.
