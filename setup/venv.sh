#!bin/bash
export PROJECT="trm-control"
cd $MYSCRATCH/${PROJECT}
rm -rf .venv
uv venv
source .venv/bin/activate
uv pip install --upgrade pip

# Modules we need to install
uv pip install torch torchvision torchaudio  --index-url https://download.pytorch.org/whl/cu126
uv pip install gymnasium imageio "gymnasium[classic-control]" "gymnasium[other]"
uv pip install wandb weave
uv pip install hydra-core --upgrade
uv pip install psutil