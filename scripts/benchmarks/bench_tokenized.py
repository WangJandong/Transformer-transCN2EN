
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

"""Compare throughput: real-time tokenization vs pre-tokenized."""
import time
import torch
import torch.nn.functional as F
from config import Config
from tokenizer import load_spm
from dataset import build_dataloaders
from dataset_tokenized import build_tokenized_dataloaders
from model import TranslationTransformer

config = Config()
config.batch_size = 256
config.num_workers = 2
config.max_train_samples = 2000
device = torch.device("cuda")

sp = load_spm(config.sp_model_prefix)
config.vocab_size = sp.get_piece_size()
V = config.vocab_size

model = TranslationTransformer(
    vocab_size=V, d_model=config.d_model, nhead=config.nhead,
    num_encoder_layers=config.num_encoder_layers,
    num_decoder_layers=config.num_decoder_layers,
    dim_feedforward=config.dim_feedforward, dropout=0.0,
    max_seq_len=config.max_seq_len, activation=config.activation,
).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
scaler = torch.amp.GradScaler("cuda")


def run_bench(dl, label):
    n_batches = 0
    total_tokens = 0
    t0 = time.perf_counter()
    for batch in dl:
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

        total_tokens += src.numel() + tgt.numel()
        n_batches += 1
        if n_batches >= 20:
            break

    elapsed = time.perf_counter() - t0
    tps = total_tokens / elapsed
    ms_per_step = elapsed / n_batches * 1000
    print(f"  {label:<30s} {tps:>10,.0f} tok/s  ({ms_per_step:.0f}ms/step)")


print("Comparing DataLoader backends …\n")

# Real-time tokenization
dl_rt, _ = build_dataloaders(config, sp)
run_bench(dl_rt, "Real-time SentencePiece")

# Pre-tokenized
dl_pt, _ = build_tokenized_dataloaders(config)
run_bench(dl_pt, "Pre-tokenized (mmap)")

print()
