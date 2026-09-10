#!/bin/bash
set -e

# -------------------------
# Module environment (Katana)
# -------------------------
# Pawsey/Setonix-specific modules removed (2026-09-07) - see setup/module.sh
# for the full explanation of what was dropped and why (Katana is a plain
# x86_64 PBS Pro cluster, not Pawsey's ARM/Cray-EX Setonix).

module purge
module add python/3.11.3   # verified working on Katana; re-check `module avail python`

# TODO(verify on Katana): no CUDA module confirmed yet from an actual GPU
# job - run `module avail cuda` on katana and pin it here before GPU training.

echo "Python: $(which python3)"
python3 -V

# -------------------------
# Virtual environment
# -------------------------
# $MYSCRATCH was Pawsey-specific; Katana's observed per-user scratch
# convention is /srv/scratch/<username> (from a real PBS job's PBS_O_PATH).

export PROJECT="trm-control"
export PROJECT_DIR="/srv/scratch/${USER}/${PROJECT}"
export UV_CACHE_DIR="$PROJECT_DIR/.uv-cache"
export TMPDIR="$PROJECT_DIR/.tmp"

mkdir -p "$UV_CACHE_DIR" "$TMPDIR"
cd "$PROJECT_DIR"

rm -rf .venv
rm -rf .tmp

uv venv --python "$(which python3)"
source .venv/bin/activate

python -V

# -------------------------
# Dependencies
# -------------------------
# Authoritative list also tracked in pyproject.toml - keep both in sync.

uv pip install --upgrade pip

uv pip install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

uv pip install \
    gymnasium \
    imageio \
    "gymnasium[classic-control]" \
    "gymnasium[other]"

uv pip install wandb weave
uv pip install --upgrade hydra-core
uv pip install psutil

echo ""
echo "Setup complete."
echo "Virtual environment: $PROJECT_DIR/.venv"
