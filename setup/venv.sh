# -------------------------
# Virtual environment (Katana)
# -------------------------
# $MYSCRATCH was a Pawsey/Setonix env var and does not exist on Katana.
# Katana's per-user scratch convention observed from an actual PBS job's
# environment (PBS_O_PATH, 2026-09-07) is /srv/scratch/<username> - adjust
# PROJECT_DIR below if your allocation differs.

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
# Kept as-is from the Pawsey setup - none of this is Pawsey-specific, it's
# plain PyPI/PyTorch-index installs, so it carries over to Katana unchanged.
# This is also the authoritative list synced into pyproject.toml - if you
# add a dependency, add it in both places.

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
