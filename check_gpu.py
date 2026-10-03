import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA runtime:", torch.version.cuda)
else:
    print("GPU training is not available in this environment. Install a CUDA-enabled PyTorch build if you have an NVIDIA GPU.")
