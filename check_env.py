"""Check available features for Turing GPU optimization."""
import torch
import torch.optim as optim

print(f"PyTorch:       {torch.__version__}")
print(f"CUDA:          {torch.version.cuda}")
print(f"cuDNN:         {torch.backends.cudnn.version()}")
print(f"GPU:           {torch.cuda.get_device_name(0)}")
print(f"Compute cap:   {torch.cuda.get_device_capability(0)}")
print(f"VRAM:          {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# torch.compile
try:
    import torch._dynamo
    print("torch.compile: AVAILABLE")
except ImportError:
    print("torch.compile: NOT AVAILABLE")

# Fused AdamW
sig = optim.AdamW.__init__.__code__.co_varnames
print(f"fused AdamW:   {'AVAILABLE' if 'fused' in sig else 'NOT AVAILABLE'}")

# SDPA backends
has_flash = torch.backends.cuda.flash_sdp_enabled()
has_mem = torch.backends.cuda.mem_efficient_sdp_enabled()
has_math = torch.backends.cuda.math_sdp_enabled()
print(f"Flash SDPA:    {'YES' if has_flash else 'NO (requires Ampere+)'}")
print(f"Mem-eff SDPA:  {'YES' if has_mem else 'NO'}")
print(f"Math SDPA:     {'YES' if has_math else 'NO'}")

# TF32
print(f"TF32 matmul:   {torch.backends.cuda.matmul.allow_tf32}")
print(f"TF32 cudnn:    {torch.backends.cudnn.allow_tf32}")

# xformers
try:
    import xformers
    print(f"xformers:      {xformers.__version__}")
except ImportError:
    print("xformers:      NOT INSTALLED")

trt_avail = False
try:
    import torch_tensorrt
    print(f"Torch-TRT:     {torch_tensorrt.__version__}")
    trt_avail = True
except ImportError:
    print("Torch-TRT:     NOT INSTALLED")

# Check Turing-specific FP16 tensor core info
print(f"\n--- Turing FP16 Tensor Core performance ---")
print(f"FP16 is 2x FP32 on Turing Tensor Cores for matmul")
print(f"Recommend: mixed precision with torch.amp(autocast)")
