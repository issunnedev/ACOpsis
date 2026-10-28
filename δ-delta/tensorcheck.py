import torch

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))
print("BF16 supported:", torch.cuda.is_bf16_supported())

x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
y = x @ x
print("BF16 test passed:", y.dtype)