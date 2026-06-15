#!/bin/bash
set -e

# -------------------------
# Module environment
# -------------------------

module purge
hash -r
module load pawsey pawseytools pawseyenv/2026.01
module reload pawseyenv/2026.01
module load PrgEnv-nvidia
module load craype-arm-grace
module load gcc-native/12.3
# module load gcc-native-mixed/12.3
module load cuda/12.6
module load python/3.11.6

export MYSCRATCH="/scratch/pawsey1302/oalmabrouk"

echo "Python: $(which python3)"
python3 -V

# -------------------------
# Virtual environment
# -------------------------

export PROJECT="trm-control"
export UV_CACHE_DIR="$PWD/.uv-cache"
export TMPDIR="$PWD/.tmp"

mkdir -p "$UV_CACHE_DIR" "$TMPDIR"

rm -rf .venv

uv venv --python "$(which python3)"
source .venv/bin/activate

python -V

# -------------------------
# Dependencies
# -------------------------

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
echo "Virtual environment: $PWD/.venv"