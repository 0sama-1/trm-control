import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))
#uv pip install torch torchvision torchaudio   --index-url https://download.pytorch.org/whl/cu126