"""Benchmark: compare training throughput with/without Turing optimizations."""
import time
import torch
import numpy as np

from config import Config
from tokenizer import load_spm
from model import TranslationTransformer
from trainer import configure_gpu_backend

config = Config()
configure_gpu_backend()

device = torch.device("cuda")
sp = load_spm(config.sp_model_prefix)
config.vocab_size = sp.get_piece_size()

# Fake data — representative shapes
B, S = 128, 128  # batch_size, seq_len
V = config.vocab_size
src = torch.randint(4, V, (B, S), device=device)
tgt = torch.randint(4, V, (B, S), device=device)
dec_input = tgt[:, :-1]
dec_label = tgt[:, 1:].reshape(-1)

warmup = 5
iters = 50
compile_mode = config.compile_mode  # "default"

print(f"\n{'='*55}")
print(f"  Benchmark: {B=}, {S=}, vocab={V}")
print(f"{'='*55}\n")

# ── Helper ──
def bench(model, label, use_amp=True, compile_mode=None, fused=True):
    model = model.to(device)
    if compile_mode and hasattr(torch, "compile"):
        model = torch.compile(model, mode=compile_mode)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=fused) if fused else \
          torch.optim.AdamW(model.parameters(), lr=1e-4)

    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    model.train()

    from contextlib import nullcontext
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()

    # Warmup
    for _ in range(warmup):
        opt.zero_grad()
        with amp_ctx:
            logits = model(src, dec_input)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, V), dec_label)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        opt.zero_grad()
        with amp_ctx:
            logits = model(src, dec_input)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, V), dec_label)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tok_per_iter = B * (S + S)  # src + tgt tokens
    tok_per_sec = tok_per_iter * iters / elapsed
    mem = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    return tok_per_sec, mem


def make_model():
    return TranslationTransformer(
        vocab_size=V,
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=0.0,  # no dropout during bench
        max_seq_len=S + 4,
        activation=config.activation,
    )

results = {}

# 1. Baseline: no AMP, no compile, no fused
m = make_model()
toks, mem = bench(m, "FP32 baseline", use_amp=False, compile_mode=None, fused=False)
results["FP32 baseline"] = (toks, mem)
print(f"  {'FP32 (baseline)':<32} {toks:>10,.0f} tok/s  ({mem:.1f} GB peak)")

# 2. AMP only (FP16 Tensor Cores)
m = make_model()
toks, mem = bench(m, "AMP FP16", use_amp=True, compile_mode=None, fused=False)
results["+ AMP"] = (toks, mem)
print(f"  {'+ AMP (fp16 tensor cores)':<32} {toks:>10,.0f} tok/s  ({mem:.1f} GB peak)")

# 3. AMP + Fused AdamW
m = make_model()
toks, mem = bench(m, "AMP + Fused", use_amp=True, compile_mode=None, fused=True)
results["+ Fused AdamW"] = (toks, mem)
print(f"  {'+ Fused AdamW':<32} {toks:>10,.0f} tok/s  ({mem:.1f} GB peak)")

# 4. AMP + Fused + torch.compile (default)
m = make_model()
toks, mem = bench(m, "AMP + Fused + compile", use_amp=True, compile_mode="default", fused=True)
results["+ torch.compile"] = (toks, mem)
print(f"  {'+ torch.compile(default)':<32} {toks:>10,.0f} tok/s  ({mem:.1f} GB peak)")

# 5. AMP + Fused + torch.compile (reduce-overhead)
m = make_model()
toks, mem = bench(m, "AMP + Fused + compile (reduce-overhead)", use_amp=True, compile_mode="reduce-overhead", fused=True)
results["+ compile(reduce-overhead)"] = (toks, mem)
print(f"  {'+ torch.compile(reduce-overhead)':<32} {toks:>10,.0f} tok/s  ({mem:.1f} GB peak)")

print(f"\n  {'─'*50}")
baseline = results["FP32 baseline"][0]
for name, (toks, mem) in results.items():
    speedup = toks / baseline
    print(f"  {name:<32} {toks:>10,.0f} tok/s  ×{speedup:.2f}")
print(f"{'='*55}")
