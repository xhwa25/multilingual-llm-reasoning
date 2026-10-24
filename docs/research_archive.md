# Private research archive

The local Research Archive is a sibling of this repository at
`../Multilingual LLM Research Archive/`. It is **outside Git**. The repository's
`.gitignore` also excludes generated `outputs/`, private `data/`, caches, and model
weights if they are restored locally.

The archive contains three top-level snapshots:

- `outputs/`: all project output files present at the time of archival, including
  historical experiments and copies of the frozen Phase 16H datasets, Phase 16I
  training manifest, provisional test grid, and real base predictions.
- `data/`: processed GSM8K subsets, SFT working files, and translation drafts.
- `final-evaluation-download/`: the downloaded Phase 16L final evaluation files,
  including the combined 1,176-row raw predictions and selected test grid.
- `adapters/english_lora/` and `adapters/multilingual_lora/`: the original LoRA
  `adapter_config.json`, `adapter_model.safetensors`, and
  `evaluation_provenance.json` files for both conditions.

`archive_file_manifest.json` lists each copied file's relative path, byte count,
and SHA-256. All 620 source files (113,146,722 bytes) were compared byte-for-byte
by hash after copying. The two frozen SFT files match their published hashes:
`6e59b5344db55eb84f87c9dc9059587fbd00dae0edcea967a76d37705b8ed881`
and `9b5c813da240229a31a7e57d1bc8a8e4cb9e8a410a837c38b01db23202df2021`.
The combined final predictions have 1,176 unique condition/problem/language keys,
392 per condition. The aggregate results copied into `docs/evidence/phase16l/`
remain the publishable evidence.

`adapter_manifest.json` records all six adapter file hashes and sizes. Each
downloaded file was copied and verified against both its source and the Phase 16L
reproducibility manifest. Both weight files contain 504 safetensors entries and
their configurations and provenance identify the expected Qwen base, adapter
condition, and pinned revision. The original Google Drive training archive should
still be retained as a separate backup. The earlier weights-only manifest and
archive guide are preserved under `adapter_manifest-weights-only.json` and
`README-before-provenance.md` rather than overwritten.

## Restore a needed file or directory

Work from the repository root and check that the destination is absent before
copying. For example, to restore the historical Phase 15 report without replacing
any current file:

```sh
archive="../Multilingual LLM Research Archive"
test ! -e outputs/phase15-baseline-analysis
cp -pR "$archive/outputs/phase15-baseline-analysis" outputs/
```

For a final-run input, use the exact relative path recorded in the corresponding
manifest and compare SHA-256 with `archive_file_manifest.json` after restoration.
Never merge an archived directory over a newer run without auditing its hashes.
The currently retained ignored directories under `outputs/` and `data/` are kept
in place because the final scripts and versioned configurations reference them.
The `.cache/` Hugging Face model snapshot and Python environments remain local and
ignored; moving them would invalidate existing local runtime paths.
