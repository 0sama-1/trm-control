import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name: ", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
