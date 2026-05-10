# Training Performance Bottleneck Analysis

**Hardware**: RTX 2080 Ti (Turing TU102, SM 7.5, 22GB VRAM)
**Date**: 2026-05-05
**Profiling tool**: PyTorch Profiler + Arithmetic Intensity Roofline

---

## 1. Theoretical Specs & Roofline

| Metric | Value |
|--------|-------|
| SMs / FP32 cores / Tensor Cores | 68 / 4352 / 544 |
| Boost clock | 1.545 GHz |
| Memory bandwidth (theoretical) | 616 GB/s (352-bit GDDR6) |
| Memory bandwidth (measured, elem-wise) | **~360 GB/s (58%)** |
| FP32 peak | 13.5 TFLOPS |
| FP16 Tensor Core peak | **107.6 TFLOPS** |
| L2 cache | 5.5 MB |

**Roofline ridge point (FP16 TC): ~175 FLOP/byte**

```
    TFLOPS
       ↑
  107.6┼─────────────────────────────╮  ← Compute bound
       │                           ╱
       │                        ╱
       │                     ╱  ← Memory bound
       │                  ╱
       │               ╱
       └─────────────┼──────────────────→ FLOP/byte
                     175
```

Operations with AI < 175 are **memory-bandwidth-bound**.
Operations with AI > 175 are **compute-bound**.

---

## 2. Arithmetic Intensity Per Operation (B=128, S=128, d=512)

| Operation | GFLOPs | MB moved | AI (FLOP/byte) | Bound |
|-----------|--------|----------|----------------|-------|
| QKV projection (in_proj) | 25.8 | 68.7 | **375** | COMPUTE |
| Attention scores QK^T | 2.2 | 100.7 | **21** | MEMORY |
| Attention output @V | 2.2 | 100.7 | **21** | MEMORY |
| Attn output projection | 8.6 | 34.1 | **252** | COMPUTE |
| FFN linear1 (d→4d) | 34.4 | 86.0 | **400** | COMPUTE |
| FFN linear2 (4d→d) | 34.4 | 86.0 | **400** | COMPUTE |
| LayerNorm | 0.03 | 16.8 | **2** | MEMORY |
| Softmax | 0.05 | 67.1 | **0.8** | MEMORY |
| Output projection (d→V) | 536.9 | 1098.1 | **489** | COMPUTE |
| **Total per layer** | **644.3** | **1658** | **389** | — |

**Key insight**: The model is **compute-bound overall** (AI=389 > ridge=175), but individual operations in the attention path (QK^T, softmax, @V, LayerNorm) are **severely memory-bound**. These "memory holes" waste GPU cycles.

---

## 3. Profiler Trace — CUDA Kernel Time Breakdown

**Per training step (B=128, S=128)**:

| Kernel / Operation | CUDA time (ms) | % | Category |
|---------------------|-----------|------|----------|
| Encoder-Decoder forward | 320.2 | 28.4% | Compute |
| GEMM kernels (`turing_fp16_s1688gemm_*`) | 216.4 | 19.2% | Compute (TC) |
| `aten::copy_` (data movement) | 176.2 | 15.6% | **Bandwidth** |
| `aten::addmm` (backward matmul) | 102.1 | 9.0% | Compute (TC) |
| Element-wise kernels | 86.9 | 7.7% | **Bandwidth** |
| FMHA CUTLASS (attention) | 77.4 | 6.9% | Mixed |
| `aten::_fused_adamw_` | 32.4 | 2.9% | Optimizer |
| Other | 32.3 | 2.9% | — |
| **Total CUDA** | **1129** | **100%** | — |

**CPU-side**:
- `cudaLaunchKernel` CPU time: 232ms (16% of CPU time) — **kernel launch overhead**

---

## 4. Bottleneck Identification

### Bottleneck 1: Memory-bandwidth-bound attention (AI=21)

QK^T and softmax(@V) together read ~200MB but only do ~4.3 GFLOPs per layer.
Measured bandwidth is only ~360 GB/s (58% of theoretical), so these ops stall the GPU.

**Mitigation**: Already using CUTLASS-based `fmha` (visible in profiler as `fmha_cutlassB_f16_aligned_64x64_k64_dropout_sm75`). This is the best available attention kernel for Turing SM 7.5 — FlashAttention-2 requires Ampere+.

### Bottleneck 2: Data movement (`aten::copy_` = 15.6%)

`aten::copy_` accounts for 176ms per step. Sources:
- `pin_memory` → GPU transfers in DataLoader
- FP32 → FP16 conversions before matmul
- Memory layout transformations

**Mitigation**: 
- `torch.compile` already fuses some copies
- Enable `channels_last` memory format for tensor layouts
- Increase `prefetch_factor` in DataLoader
- Reduce `num_workers` if CPU→GPU transfer is the bottleneck (avoid contention)

### Bottleneck 3: Kernel launch overhead (16% CPU time)

232ms of CPU time spent in `cudaLaunchKernel`. On Turing, each kernel launch is ~6-10µs. With 4500 kernel launches per step, the overhead adds up.

**Mitigation**:
- `torch.compile(mode="reduce-overhead")` uses CUDA graphs to batch launches
- Already configured in current defaults

### Bottleneck 4: Element-wise operations

86.9ms (7.7%) in element-wise kernels (ReLU, dropout, residual add, mask application). These are pure memory-bandwidth operations.

**Mitigation**: `torch.compile`'s inductor backend already fuses element-wise ops with preceding matmuls where possible (confirmed by `turing_fp16_s1688gemm_fp16_256x128_ldg8_relu_f2f_tn` — ReLU is fused into the GEMM kernel).

---

## 5. Optimisation Effectiveness Summary

| Optimization | Applied? | Speedup | Notes |
|-------------|----------|---------|-------|
| FP16 Tensor Cores (AMP) | ✅ | ×2.69 | Single biggest win |
| torch.compile (default) | ✅ | +7% | Kernel fusion, reduces element-wise overhead |
| CUTLASS FMHA (attention) | ✅ built-in | — | Best available for Turing SM 7.5 |
| Fused AdamW | ✅ | minimal | Small model, optimizer is ~3% of step |
| TF32 disabled | ✅ | avoids slowdown | Turing has no native TF32 |
| CUDA graphs | ❌ (default mode) | +0% | Could add 3-5% but conflicts with GradScaler |

---

## 6. Actionable Recommendations

| Priority | Action | Expected gain |
|----------|--------|---------------|
| **P0** | Increase batch_size to 256-512 (22GB VRAM has room) | +10-20% (better GPU utilisation) |
| **P1** | Enable `channels_last` memory format | -5-10% data movement overhead |
| **P2** | Tune `num_workers` (try 2 vs 4 vs 8) | Reduce DataLoader stalls |
| **P2** | Try `torch.compile(mode="max-autotune")` | +3-5% (coordinate descent on matmul configs) |
| **P3** | Profile DataLoader as separate bottleneck | Ensure GPU is not starved |
| **P3** | Consider gradient accumulation > 2 | Larger effective batch for smoother gradients |
