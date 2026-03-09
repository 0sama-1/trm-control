#!bin/bash
export PROJECT="trm-control"
cd $MYSCRATCH/${PROJECT}
rm -rf .venv
uv venv
source .venv/bin/activate
uv pip install --upgrade pip
uv pip install torch torchvision torchaudio  --index-url https://download.pytorch.org/whl/cu126