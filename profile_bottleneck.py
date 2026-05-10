"""Comprehensive performance bottleneck analysis for NMT training on Turing.

Covers:
  1. Theoretical GPU specs & roofline model
  2. PyTorch Profiler trace (kernel-level hotspots)
  3. Arithmetic-intensity breakdown per operation
  4. Memory-bandwidth utilisation analysis
"""
from __future__ import annotations
import math
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, record_function, schedule

from config import Config
from tokenizer import load_spm, BOS_ID, EOS_ID
from model import TranslationTransformer
from trainer import configure_gpu_backend


# ═══════════════════════════════════════════════════════════════════
# 1. THEORETICAL SPECS
# ═══════════════════════════════════════════════════════════════════

def print_theoretical_specs():
    prop = torch.cuda.get_device_properties(0)
    # Turing TU102 figures
    sm_count = prop.multi_processor_count  # 68
    # Each SM: 64 FP32 CUDA cores + 8 Tensor Cores
    boost_clock = 1.545  # GHz (typical 2080 Ti boost)
    mem_bw = 616.0       # GB/s (352-bit GDDR6 @ 14 Gbps)
    l2_cache = 5.5       # MB

    fp32_per_sm = 64 * 2 * boost_clock  # 64 cores × 2 FMA/clock × clock
    fp32_theo = fp32_per_sm * sm_count   # TFLOPS
    # Turing Tensor Core: 64 FP16 FMA/clock per TC, 8 TCs per SM
    fp16_tc_per_sm = 8 * 64 * 2 * boost_clock
    fp16_tc_theo = fp16_tc_per_sm * sm_count

    print(f"{'='*60}")
    print(f"  THEORETICAL SPECS — RTX 2080 Ti (Turing TU102)")
    print(f"{'='*60}")
    print(f"  SMs: {sm_count}  |  L2: {l2_cache} MB  |  Boost: {boost_clock} GHz")
    print(f"  Memory bandwidth: {mem_bw:.0f} GB/s")
    print(f"  FP32 peak:  {fp32_theo:.1f} TFLOPS")
    print(f"  FP16 peak:  {fp32_theo*2:.1f} TFLOPS  (CUDA cores, no TC)")
    print(f"  FP16 TC peak: {fp16_tc_theo:.1f} TFLOPS  (Tensor Cores)")
    print(f"  Ridge point (FP16 TC): {fp16_tc_theo*1000/mem_bw:.0f} FLOP/byte")
    print(f"  Ridge point (FP32):    {fp32_theo*1000/mem_bw:.0f} FLOP/byte")

    return {
        "fp16_tc_tflops": fp16_tc_theo,
        "fp32_tflops": fp32_theo,
        "mem_bw": mem_bw,
        "ridge_fp16": fp16_tc_theo * 1000 / mem_bw,
        "ridge_fp32": fp32_theo * 1000 / mem_bw,
    }


# ═══════════════════════════════════════════════════════════════════
# 2. ARITHMETIC INTENSITY PER OPERATION
# ═══════════════════════════════════════════════════════════════════

def analyse_arithmetic_intensity(config):
    d = config.d_model       # 512
    df = config.dim_feedforward  # 2048
    V = config.vocab_size    # 32000
    nhead = config.nhead     # 8
    B = 128
    S = 128  # representative sequence length
    d_head = d // nhead      # 64

    print(f"\n{'='*60}")
    print(f"  ARITHMETIC INTENSITY ANALYSIS  (B={B}, S={S}, d={d}, d_ff={df})")
    print(f"{'='*60}")
    print(f"  Ridge point (FP16 TC): ~{config._ridge_fp16:.0f} FLOP/byte")
    print(f"  Below ridge → memory-bound    Above ridge → compute-bound")
    print()

    ops = []

    # -- QKV projection (in_proj): (B×S, d) × (d, 3d) --
    # FLOPs: 2 * B*S * d * 3d = 6 * B*S * d^2
    linear_flops = 2 * B * S * d * d  # generic: 2*M*K*N for (M,K)×(K,N)
    qkv_flops = 3 * linear_flops
    qkv_bytes = (B * S * d + 3 * d * d + 3 * B * S * d) * 2  # fp16: ×2 bytes
    ops.append(("QKV projection", qkv_flops, qkv_bytes))

    # -- Attention scores QK^T: (B,n,S,dh) × (B,n,dh,S) → (B,n,S,S) --
    attn_scores_flops = 2 * B * nhead * S * S * d_head
    attn_scores_bytes = (B * nhead * S * d_head * 2 + B * nhead * S * S * 2) * 2
    ops.append(("Attention scores QK^T", attn_scores_flops, attn_scores_bytes))

    # -- Attention output: softmax(QK^T) × V --
    attn_out_flops = 2 * B * nhead * S * S * d_head
    attn_out_bytes = (B * nhead * S * S * 2 + B * nhead * S * d_head * 2) * 2
    ops.append(("Attention output @V", attn_out_flops, attn_out_bytes))

    # -- Attention out_proj: (B×S, d) × (d, d) --
    out_proj_flops = 2 * B * S * d * d
    out_proj_bytes = (B * S * d + d * d + B * S * d) * 2
    ops.append(("Attn out projection", out_proj_flops, out_proj_bytes))

    # -- FFN layer1: (B×S, d) × (d, d_ff) --
    ffn1_flops = 2 * B * S * d * df
    ffn1_bytes = (B * S * d + d * df + B * S * df) * 2
    ops.append(("FFN linear1 (up)", ffn1_flops, ffn1_bytes))

    # -- FFN layer2: (B×S, d_ff) × (d_ff, d) --
    ffn2_flops = 2 * B * S * df * d
    ffn2_bytes = (B * S * df + df * d + B * S * d) * 2
    ops.append(("FFN linear2 (down)", ffn2_flops, ffn2_bytes))

    # -- LayerNorm --
    ln_flops = 4 * B * S * d  # mean, variance, normalize, scale
    ln_bytes = (B * S * d + 2 * d) * 2  # input + weight+bias
    ops.append(("LayerNorm", ln_flops, ln_bytes))

    # -- Output projection: (B×S, d) × (d, V) --
    out_flops = 2 * B * S * d * V
    out_bytes = (B * S * d + d * V + B * S * V) * 2
    ops.append(("Output projection", out_flops, out_bytes))

    # -- Softmax (attention) --
    softmax_flops = 3 * B * nhead * S * S  # exp + sum + divide
    softmax_bytes = 2 * B * nhead * S * S * 2  # in + out
    ops.append(("Softmax (attn)", softmax_flops, softmax_bytes))

    print(f"  {'Operation':<25s} {'GFLOPs':>8s} {'MB moved':>9s} {'AI':>8s}  {'Bound'}")
    print(f"  {'-'*65}")
    total_flops = 0
    total_bytes = 0
    for name, flops, byt in ops:
        ai = flops / max(byt, 1)
        bound = "COMPUTE" if ai > 60 else "MEMORY"
        total_flops += flops
        total_bytes += byt
        print(f"  {name:<25s} {flops/1e9:8.2f} {byt/1e6:9.1f} {ai:8.1f}  {bound}")

    total_ai = total_flops / max(total_bytes, 1)
    print(f"  {'─'*65}")
    print(f"  {'TOTAL (1 layer)':<25s} {total_flops/1e9:8.2f} {total_bytes/1e6:9.1f} {total_ai:8.1f}")
    print(f"  {'TOTAL (×12 layers)':<25s} {total_flops*12/1e9:8.2f}")
    print(f"  {'TOTAL (×12+output)':<25s} {total_flops*12/1e9+out_flops/1e9:8.2f}")

    return total_ai


# ═══════════════════════════════════════════════════════════════════
# 3. PYTORCH PROFILER
# ═══════════════════════════════════════════════════════════════════

def run_profiler(config, specs):
    device = torch.device("cuda")
    sp = load_spm(config.sp_model_prefix)
    config.vocab_size = sp.get_piece_size()

    model = TranslationTransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len,
        activation=config.activation,
    ).to(device)

    # torch.compile is already in the model — skip for profiling purity
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scaler = torch.amp.GradScaler("cuda")

    B, S = 128, 128
    V = config.vocab_size
    src = torch.randint(4, V, (B, S), device=device)
    tgt = torch.randint(4, V, (B, S), device=device)

    print(f"\n{'='*60}")
    print(f"  PYTORCH PROFILER TRACE")
    print(f"{'='*60}")

    # Warmup
    model.train()
    for _ in range(5):
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(src, tgt[:, :-1])
            loss = criterion(logits.reshape(-1, V), tgt[:, 1:].reshape(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    torch.cuda.synchronize()

    # Profiled steps
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities, record_shapes=True, with_stack=True,
                 schedule=schedule(wait=2, warmup=2, active=3, repeat=1)) as prof:
        for step in range(9):
            with record_function(f"train_step_{step}"):
                optimizer.zero_grad()
                with record_function("forward"):
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        logits = model(src, tgt[:, :-1])
                        loss = criterion(logits.reshape(-1, V), tgt[:, 1:].reshape(-1))
                with record_function("backward"):
                    scaler.scale(loss).backward()
                with record_function("optimizer"):
                    scaler.step(optimizer)
                    scaler.update()
            prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))

    # Also do a memory bandwidth utilisation check
    print(f"\n{'='*60}")
    print(f"  MEMORY BANDWIDTH UTILISATION")
    print(f"{'='*60}")

    total_cuda_time_us = 0
    total_bytes_transferred = 0
    dram_bytes = 0
    for evt in prof.key_averages():
        if evt.cuda_time_total > 0:
            total_cuda_time_us += evt.cuda_time_total

    # Rough estimate from profiling
    print(f"  Total CUDA time: {total_cuda_time_us/1e6:.3f} s")
    print(f"  Peak memory allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # Bandwidth stress test
    print(f"\n  ── Bandwidth saturation test ──")
    for size_mb in [16, 64, 256, 1024]:
        n = size_mb * 1024 * 1024 // 4  # float32 elements
        x = torch.randn(1, n, device=device)
        # Measure read + write bandwidth
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            y = x * 2.0 + 1.0  # element-wise: pure memory-bound
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        byt_per_iter = x.numel() * 4 * 3  # read x, write y, read consts
        bw_achieved = byt_per_iter * 100 / elapsed / 1e9
        print(f"    {size_mb:5d} MB:  {bw_achieved:6.1f} GB/s  "
              f"({bw_achieved/specs['mem_bw']*100:5.1f}% of theoretical {specs['mem_bw']:.0f} GB/s)")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    configure_gpu_backend()
    config = Config()
    specs = print_theoretical_specs()
    config._ridge_fp16 = specs["ridge_fp16"]  # stash for AI analysis
    ai = analyse_arithmetic_intensity(config)
    run_profiler(config, specs)
