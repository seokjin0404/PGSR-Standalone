#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${PGSR_VENV:-$REPO_ROOT/.venv}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python not found: $PYTHON_BIN" >&2
    exit 1
fi
if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc not found. Install the CUDA toolkit first." >&2
    exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
PYTHON="$VENV_DIR/bin/python"
"$PYTHON" -m pip install --upgrade pip setuptools wheel ninja
if ! "$PYTHON" -c 'import torch, torchvision' >/dev/null 2>&1; then
    "$PYTHON" -m pip install torch torchvision
fi
"$PYTHON" -m pip install -r "$REPO_ROOT/requirements.txt"
"$PYTHON" -m pip install --no-build-isolation \
    "$REPO_ROOT/submodules/diff-plane-rasterization"
"$PYTHON" -m pip install --no-build-isolation \
    "$REPO_ROOT/submodules/simple-knn"
"$PYTHON" -c 'import torch, diff_plane_rasterization, simple_knn; print(f"PGSR ready: torch={torch.__version__}, CUDA={torch.version.cuda}, GPUs={torch.cuda.device_count()}")'

echo "Setup complete. Run:"
echo "  1. Change only DATA_ROOT in scripts/run_pgsr_synthetic.sh and scripts/run_pgsr_real.sh"
echo "  2. bash scripts/run_pgsr_synthetic.sh"
echo "  3. bash scripts/run_pgsr_real.sh"
