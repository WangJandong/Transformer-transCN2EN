"""Test torch.compile backends — catch runtime crashes, not just compile errors."""
import time, torch, torch.nn.functional as F
from config import Config; from tokenizer import load_spm
from dataset_tokenized import build_tokenized_dataloaders
from model import TranslationTransformer; from trainer import configure_gpu_backend

configure_gpu_backend()
sp = load_spm("spm_bpe"); V = sp.get_piece_size()
cfg = Config(); cfg.batch_size = 128; cfg.max_train_samples = 2000
cfg.num_workers = 0
device = torch.device("cuda")

def bench(label, compile_mode=None, backend=None):
    m = TranslationTransformer(
        vocab_size=V, d_model=cfg.d_model, nhead=cfg.nhead,
        num_encoder_layers=6, num_decoder_layers=6,
        dim_feedforward=cfg.dim_feedforward, dropout=0.0,
        max_seq_len=cfg.max_seq_len, activation=cfg.activation,
    ).to(device)

    kwargs = {}
    if compile_mode: kwargs["mode"] = compile_mode
    if backend: kwargs["backend"] = backend
    if kwargs:
        m = torch.compile(m, **kwargs)

    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)
    scaler = torch.amp.GradScaler("cuda")
    m.train()

    tl, _, _ = build_tokenized_dataloaders(cfg)
    total_tok = 0

    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i, batch in enumerate(tl):
            src, tgt, _, _ = [b.to(device, non_blocking=True) for b in batch]
            di, dl = tgt[:, :-1], tgt[:, 1:]
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = m(src, di)
                loss = F.cross_entropy(logits.reshape(-1, V), dl.reshape(-1),
                                       ignore_index=0, label_smoothing=0.1)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad()
            total_tok += src.numel() + tgt.numel()
            if i >= 20: break
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        tps = total_tok / elapsed
        print(f"  {label:<45s} {tps:>10,.0f} tok/s")
        return tps
    except Exception as e:
        msg = str(e)[:100].replace("\n", " ")
        print(f"  {label:<45s} CRASH: {msg}")
        return None

print(f"\n{'='*65}")
print(f"  torch.compile BACKEND SWEEP — Turing SM 7.5, real data")
print(f"{'='*65}\n")

base = bench("no compile")
print()

# inductor variants (may crash with sympy bug)
bench("inductor, mode=default")
bench("inductor, mode=reduce-overhead")

# safe backends (no inductor codegen)
bench("backend=eager")
bench("backend=aot_eager")
bench("backend=cudagraphs")

print(f"\n{'='*65}")
print(f"  Fastest working: see above")
print(f"{'='*65}")
