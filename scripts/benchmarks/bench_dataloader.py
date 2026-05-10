
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

"""Profile DataLoader to find CPU-GPU balance point."""
import time
from config import Config
from tokenizer import load_spm
from dataset import build_dataloaders, ParallelIterableDataset, collate_fn
import torch

config = Config()
config.batch_size = 256
config.num_workers = 4
sp = load_spm(config.sp_model_prefix)
config.vocab_size = sp.get_piece_size()

# 1. Measure pure data loading throughput (no model)
for nw in [0, 1, 2, 4, 8]:
    config.num_workers = nw
    dl, _ = build_dataloaders(config, sp)
    n_batches = 0
    total_tokens = 0
    t0 = time.perf_counter()
    for batch in dl:
        src, tgt, _, _ = batch
        total_tokens += src.numel() + tgt.numel()
        n_batches += 1
        if n_batches >= 50:
            break
    elapsed = time.perf_counter() - t0
    print(f"  num_workers={nw}: {total_tokens/elapsed:,.0f} tok/s  ({n_batches} batches in {elapsed:.1f}s)")

# 2. Check if tokenization is the bottleneck
print("\n  ── Tokenization time ──")
sp2 = load_spm(config.sp_model_prefix)
lines = []
with open("data/train.zh") as fz, open("data/train.en") as fe:
    for i, (z, e) in enumerate(zip(fz, fe)):
        if i >= 5000:
            break
        lines.append((z.strip(), e.strip()))

t0 = time.perf_counter()
for zh, en in lines:
    sp2.encode(zh, out_type=int)
    sp2.encode(en, out_type=int)
elapsed = time.perf_counter() - t0
print(f"  Tokenized {len(lines):,} pairs in {elapsed:.2f}s → {len(lines)/elapsed:,.0f} pairs/s")
print(f"  At this rate for 18M pairs: {18e6 / (len(lines)/elapsed) / 3600:.1f}h just for tokenization")

# 3. GPU idle time test — run with model
print("\n  ── GPU utilisation check ──")
from model import TranslationTransformer
device = torch.device("cuda")
model = TranslationTransformer(
    vocab_size=config.vocab_size, d_model=config.d_model, nhead=config.nhead,
    num_encoder_layers=config.num_encoder_layers, num_decoder_layers=config.num_decoder_layers,
    dim_feedforward=config.dim_feedforward, dropout=0.0,
    max_seq_len=config.max_seq_len, activation=config.activation,
).to(device)

config.num_workers = 4
dl, _ = build_dataloaders(config, sp)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
scaler = torch.amp.GradScaler("cuda")
crit = torch.nn.CrossEntropyLoss(ignore_index=0)

model.train()
batch_times = []
data_times = []
gpu_times = []

t_wait = time.perf_counter()
for i, batch in enumerate(dl):
    t_data_done = time.perf_counter()
    data_times.append(t_data_done - t_wait)

    src, tgt, _, _ = [b.to(device) for b in batch]
    dec_input, dec_label = tgt[:, :-1], tgt[:, 1:]

    torch.cuda.synchronize()
    t_gpu_start = time.perf_counter()
    with torch.amp.autocast("cuda", dtype=torch.float16):
        logits = model(src, dec_input)
        loss = crit(logits.reshape(-1, config.vocab_size), dec_label.reshape(-1))
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    opt.zero_grad()
    torch.cuda.synchronize()
    t_gpu_end = time.perf_counter()

    gpu_times.append(t_gpu_end - t_gpu_start)
    batch_times.append(t_gpu_end - t_wait)
    t_wait = time.perf_counter()

    if i >= 40:
        break

import numpy as np
dt = np.array(data_times)
gt = np.array(gpu_times)
print(f"  Data wait time (avg): {dt.mean()*1000:.1f}ms")
print(f"  GPU step time (avg):  {gt.mean()*1000:.1f}ms")
print(f"  GPU idle ratio:       {dt.mean()/(dt.mean()+gt.mean())*100:.1f}%")
print(f"  GPU busy ratio:       {gt.mean()/(dt.mean()+gt.mean())*100:.1f}%")
