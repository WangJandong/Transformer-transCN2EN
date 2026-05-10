"""Compare throughput across configurations and estimate total training time."""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from tokenizer import load_spm
from model import TranslationTransformer
from trainer import configure_gpu_backend

configure_gpu_backend()
config = Config()
device = torch.device("cuda")

sp = load_spm(config.sp_model_prefix)
config.vocab_size = sp.get_piece_size()

V = config.vocab_size

def make_model():
    return TranslationTransformer(
        vocab_size=V, d_model=config.d_model, nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        dim_feedforward=config.dim_feedforward, dropout=0.0,
        max_seq_len=config.max_seq_len, activation=config.activation,
    )


def bench_config(label, B, compile_mode="default", use_amp=True):
    """Measure step time with random data at given batch size."""
    # Use representative sequence length ~50 (close to real data avg=36 + padding)
    S = 50
    src = torch.randint(4, V, (B, S), device=device)
    tgt = torch.randint(4, V, (B, S), device=device)

    model = make_model().to(device)
    if compile_mode and hasattr(torch, "compile"):
        model = torch.compile(model, mode=compile_mode)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    crit = nn.CrossEntropyLoss(ignore_index=0)

    model.train()

    # Warmup
    for _ in range(10):
        opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(src, tgt[:, :-1])
            loss = crit(logits.reshape(-1, V), tgt[:, 1:].reshape(-1))
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_steps = 30
    for _ in range(n_steps):
        opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(src, tgt[:, :-1])
            loss = crit(logits.reshape(-1, V), tgt[:, 1:].reshape(-1))
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    step_ms = elapsed / n_steps * 1000
    tok_per_step = B * (S + S)  # src + tgt
    tok_per_sec = tok_per_step * n_steps / elapsed
    mem = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    return step_ms, tok_per_sec, mem


print(f"{'='*65}")
print(f"  BATCH SIZE SWEEP  (S≈50, compiled, AMP fp16, fused AdamW)")
print(f"{'='*65}")
print(f"  {'Config':<35s} {'step_ms':>8s} {'tok/s':>10s} {'VRAM':>8s}")
print(f"  {'-'*55}")

results = []
for B in [128, 192, 256, 320, 384, 512]:
    try:
        step_ms, tps, mem = bench_config(f"B={B}", B, compile_mode="default")
        results.append((B, step_ms, tps, mem))
        print(f"  B={B:<4d}  compile=default               {step_ms:8.1f}  {tps:>10,.0f}  {mem:7.1f}G")
    except torch.cuda.OutOfMemoryError:
        print(f"  B={B:<4d}  compile=default               {'OOM':>8s}")
        break

# Also test compile=reduce-overhead at best batch size
print()
best_B = max(results, key=lambda x: x[2])[0] if results else 256
try:
    step_ms, tps, mem = bench_config(f"B={best_B} reduce-oh", best_B, compile_mode="reduce-overhead")
    results.append((f"{best_B}(r)", step_ms, tps, mem))
    print(f"  B={best_B:<4d}  compile=reduce-overhead      {step_ms:8.1f}  {tps:>10,.0f}  {mem:7.1f}G")
except Exception:
    pass

# ── Estimate total training time ──
print(f"\n{'='*65}")
print(f"  TOTAL TRAINING TIME ESTIMATE")
print(f"{'='*65}")

# Real data stats from earlier analysis
TRAIN_PAIRS = 18_753_893   # after filtering
AVG_TOKENS_PER_PAIR = 36   # src+tgt BPE tokens

for row in results:
    if isinstance(row[0], str):
        continue
    B, step_ms, tps, mem = row
    # With real dynamic padding, throughput is ~10-15% lower due to padding waste
    # and variable-length overhead
    real_tps = tps * 0.85

    tok_per_epoch = TRAIN_PAIRS * AVG_TOKENS_PER_PAIR
    sec_per_epoch = tok_per_epoch / real_tps
    hrs_per_epoch = sec_per_epoch / 3600
    total_hrs_20 = hrs_per_epoch * 20

    print(f"\n  B={B}:")
    print(f"    Benchmark throughput:  {tps:,.0f} tok/s (S=50 fixed)")
    print(f"    Est. real throughput:  {real_tps:,.0f} tok/s (dynamic pad, ×0.85)")
    print(f"    Tokens per epoch:      {tok_per_epoch/1e9:.2f}B")
    print(f"    Hours per epoch:       {hrs_per_epoch:.1f}h")
    print(f"    Total (20 epochs):     {total_hrs_20:.0f}h  ({total_hrs_20/24:.1f} days)")

print(f"\n{'='*65}")
