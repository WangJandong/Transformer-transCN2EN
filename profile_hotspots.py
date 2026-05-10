"""Fine-grained profiling of a real training step — identify every hotspot."""
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, record_function
from config import Config
from tokenizer import load_spm
from dataset_tokenized import build_tokenized_dataloaders
from model import TranslationTransformer
from trainer import configure_gpu_backend

configure_gpu_backend()
device = torch.device("cuda")

sp = load_spm("spm_bpe")
V = sp.get_piece_size()

cfg = Config()
cfg.batch_size = 128
cfg.max_train_samples = 2000
tl, vl, total = build_tokenized_dataloaders(cfg)
cfg._steps_per_epoch = total // cfg.batch_size

model = TranslationTransformer(
    vocab_size=V, d_model=cfg.d_model, nhead=cfg.nhead,
    num_encoder_layers=cfg.num_encoder_layers,
    num_decoder_layers=cfg.num_decoder_layers,
    dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout,
    max_seq_len=cfg.max_seq_len, activation=cfg.activation,
).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
scaler = torch.amp.GradScaler("cuda")
model.train()

# Warmup
print("Warmup ...")
for i, batch in enumerate(tl):
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
    if i >= 5:
        break

# Profile 3 steps
print("Profiling ...")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True, with_stack=False) as prof:
    for i in range(3):
        with record_function(f"step_{i}"):
            src, tgt, _, _ = [b.to(device) for b in batch]
            dec_input, dec_label = tgt[:, :-1], tgt[:, 1:]
            with record_function("forward"):
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(src, dec_input)
                    loss = F.cross_entropy(logits.reshape(-1, V), dec_label.reshape(-1),
                                           ignore_index=0, label_smoothing=0.1)
            with record_function("backward"):
                scaler.scale(loss).backward()
            with record_function("optimizer"):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

# Print breakdown by category
print("\n=== CUDA time by kernel category ===\n")
table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)
print(table)

# Aggregate into categories
total_cuda = sum(e.cuda_time_total for e in prof.key_averages())
cats = {"gemm": 0, "attention": 0, "elementwise": 0, "copy": 0,
        "adamw": 0, "norm": 0, "other": 0}
for evt in prof.key_averages():
    name = evt.key.lower()
    ct = evt.cuda_time_total / 1e3  # us → ms
    if "gemm" in name or "addmm" in name or "linear" in name:
        cats["gemm"] += ct
    elif "fmha" in name or "attention" in name or "scaled_dot" in name:
        cats["attention"] += ct
    elif "elementwise" in name or "copy" in name:
        cats["copy"] += ct
    elif "adamw" in name or "adam" in name:
        cats["adamw"] += ct
    elif "norm" in name or "layer_norm" in name:
        cats["norm"] += ct
    elif "copy" in name:
        cats["copy"] += ct
    else:
        cats["other"] += ct

print(f"\n=== Aggregated by category (ms) ===")
for k, v in sorted(cats.items(), key=lambda x: -x[1]):
    print(f"  {k:<15s} {v:8.1f} ms  ({v/total_cuda*1e3:5.1f}%)")

# GPU SM efficiency check
print(f"\n=== GPU efficiency ===")
print(f"  SM clock: {torch.cuda.clock_rate/1e6:.0f} MHz")
print(f"  Memory clock: {torch.cuda.memory_clock_rate/1e6:.0f} MHz")
print(f"  Peak mem alloc: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
