#!/usr/bin/env bash
# Isolated Linux x86_64/Python 3.11 environment; never modifies notebook or Mac packages.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo 'This installer is for Linux x86_64 Colab GPUs. Use .[dev,data,sft] on your Mac.' >&2
  exit 2
fi
python3 -m pip install --target .colab-bootstrap --upgrade 'uv==0.9.5'
UV_BIN="$PWD/.colab-bootstrap/bin/uv"
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.cache/python"
export UV_NATIVE_TLS=true
"$UV_BIN" python install 3.11.14
if [[ ! -x .venv-colab/bin/python ]]; then
  "$UV_BIN" venv --python 3.11.14 --managed-python .venv-colab
fi
"$UV_BIN" pip sync --python .venv-colab/bin/python --require-hashes --only-binary :all: --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match requirements/colab.lock
"$UV_BIN" pip install --python .venv-colab/bin/python --no-deps --no-build-isolation -e .
"$UV_BIN" pip check --python .venv-colab/bin/python
.venv-colab/bin/python -c 'import torch, transformers, peft, trl, accelerate, datasets, bitsandbytes; print("Imports OK; CUDA available:", torch.cuda.is_available())'
"$UV_BIN" pip freeze --python .venv-colab/bin/python > .venv-colab/environment.freeze.txt
