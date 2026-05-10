
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

"""Measure impact of non_blocking transfers + persistent workers."""
import time
import torch
import torch.nn.functional as F
from config import Config
from tokenizer import load_spm
from dataset_tokenized import build_tokenized_dataloaders
from model import TranslationTransformer
from trainer import configure_gpu_backend

configure_gpu_backend()
device = torch.device("cuda")

sp = load_spm("spm_bpe")
V = sp.get_piece_size()

for label, nb, nw in [("sync  (before)", False, 2), ("async (after) ", True, 2)]:
    cfg = Config()
    cfg.batch_size = 128
    cfg.max_train_samples = 2000
    cfg.num_workers = nw
    tl, vl, total = build_tokenized_dataloaders(cfg)

    model = TranslationTransformer(
        vocab_size=V, d_model=cfg.d_model, nhead=cfg.nhead,
        num_encoder_layers=cfg.num_encoder_layers,
        num_decoder_layers=cfg.num_decoder_layers,
        dim_feedforward=cfg.dim_feedforward, dropout=0.0,
        max_seq_len=cfg.max_seq_len, activation=cfg.activation,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    scaler = torch.amp.GradScaler("cuda")

    model.train()
    steps = 50
    total_tok = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i, batch in enumerate(tl):
        src, tgt, _, _ = [b.to(device, non_blocking=nb) for b in batch]
        dec_input, dec_label = tgt[:, :-1], tgt[:, 1:]
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(src, dec_input)
            loss = F.cross_entropy(logits.reshape(-1, V), dec_label.reshape(-1),
                                   ignore_index=0, label_smoothing=0.1)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        total_tok += src.numel() + tgt.numel()
        if i >= steps:
            break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    tps = total_tok / elapsed
    print(f"  {label} non_blocking={nb}: {tps:,.0f} tok/s")
