"""Compare speed/quality across model sizes in a single 1-epoch sweep.

Runs 3 configurations back-to-back, each for 1 epoch on a small subset,
and prints tok/s + final loss so you can pick the best tradeoff.

Usage:
    python bench_speed_vs_quality.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn.functional as F
from config import Config
from tokenizer import load_spm
from dataset_tokenized import build_tokenized_dataloaders
from model import TranslationTransformer
from trainer import configure_gpu_backend
import time

configure_gpu_backend()
device = torch.device("cuda")

sp = load_spm("spm_bpe")
V = sp.get_piece_size()

# Test 3 model sizes on 50K training pairs (fast ~3 min each)
SUBSET = 50000

configs = [
    # (d_model, nhead, dim_ff, enc_layers, dec_layers, label)
    (512, 8,  2048, 6, 6, "baseline d=512 L=6"),
    (384, 6,  1536, 6, 6, "smaller  d=384 L=6"),
    (384, 6,  1536, 4, 4, "shallow d=384 L=4"),
    (256, 4,  1024, 6, 6, "tiny    d=256 L=6"),
    (256, 4,  1024, 4, 4, "micro   d=256 L=4"),
]

results = []

for d, nh, dff, n_enc, n_dec, label in configs:
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")

    cfg = Config()
    cfg.vocab_size = V
    cfg.batch_size = 128
    cfg.max_seq_len = 128
    cfg.max_train_samples = SUBSET
    cfg.epochs = 1
    cfg.log_interval = 20

    model = TranslationTransformer(
        vocab_size=V, d_model=d, nhead=nh,
        num_encoder_layers=n_enc, num_decoder_layers=n_dec,
        dim_feedforward=dff, dropout=0.0,
        max_seq_len=128, activation="relu",
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params/1e6:.1f}M")

    tl, vl, total = build_tokenized_dataloaders(cfg)
    cfg._steps_per_epoch = total // cfg.batch_size

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    scaler = torch.amp.GradScaler("cuda")
    model.train()

    losses = []
    t0 = time.perf_counter()
    step = 0
    for batch in tl:
        step += 1
        src, tgt, _, _ = [b.to(device) for b in batch]
        dec_input, dec_label = tgt[:, :-1], tgt[:, 1:]

        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(src, dec_input)
            loss = F.cross_entropy(logits.reshape(-1, V), dec_label.reshape(-1),
                                   ignore_index=0, label_smoothing=0.1)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()

        losses.append(loss.item())
        if step % cfg.log_interval == 0:
            elapsed = time.perf_counter() - t0
            tok = step * cfg.batch_size * 57  # avg tok/pair
            print(f"  step {step:4d} | loss {sum(losses[-10:])/10:.4f} | {tok/elapsed:,.0f} tok/s")

    elapsed = time.perf_counter() - t0
    tok_total = step * cfg.batch_size * 57
    tps = tok_total / elapsed
    final_loss = sum(losses[-5:]) / 5 if len(losses) >= 5 else losses[-1]

    results.append((label, d, n_enc, n_dec, n_params/1e6, tps, final_loss))
    print(f"  => {tps:,.0f} tok/s | final loss {final_loss:.4f}")

# ── Summary ──
print(f"\n{'='*65}")
print(f"  SPEED vs QUALITY COMPARISON")
print(f"{'='*65}")
print(f"  {'Config':<25s} {'Params':>6s} {'tok/s':>10s} {'final loss':>10s} {'vs base':>8s}")
print(f"  {'-'*60}")
base_loss = results[0][-1]
base_tps = results[0][-2]
for label, d, ne, nd, params, tps, loss in results:
    speedup = tps / base_tps
    loss_delta = loss - base_loss
    print(f"  {label:<25s} {params:5.1f}M {tps:>10,.0f} {loss:>10.4f} {loss_delta:>+8.4f}  ×{speedup:.1f}")
print(f"{'='*65}")
