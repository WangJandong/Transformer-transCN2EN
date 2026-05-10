"""
Profile training: identify compute vs bandwidth bottlenecks.
Usage: python profile_train.py
Output: ./profiler_logs/ (for tensorboard), terminal summary
"""
import time
import sys
sys.path.insert(0, "/home/wjd/project/train")

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

from config import Config
from tokenizer import build_tokenizer_if_needed
from dataset import build_dataloaders
from model import TranslationTransformer
from trainer import configure_gpu_backend


def print_separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def profile_run():
    configure_gpu_backend()

    # ── minimal config ──────────────────────────────────────
    config = Config()
    config.max_train_samples = 2000  # tiny subset for profiling
    config.batch_size = 32
    config.d_model = 512
    config.nhead = 8
    config.num_encoder_layers = 6
    config.num_decoder_layers = 6
    config.dim_feedforward = 2048
    config.compile_mode = ""         # no compile first
    config.fused_adamw = True
    config.mixed_precision = True
    config.grad_accum_steps = 1
    config.device = "cuda"
    config.num_workers = 2

    device = torch.device("cuda")

    # ── tokenizer + data ────────────────────────────────────
    sp = build_tokenizer_if_needed(config)
    config.vocab_size = sp.get_piece_size()
    print(f"vocab_size = {config.vocab_size}")

    train_loader, _ = build_dataloaders(config, sp)

    # ── model ───────────────────────────────────────────────
    model = TranslationTransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=0.1,
        max_seq_len=config.max_seq_len,
        activation="relu",
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.1f}M params")

    # ── optimizer ───────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9, fused=True
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scaler = torch.amp.GradScaler("cuda")
    amp_dtype = torch.float16

    model.train()

    # ── warmup ──────────────────────────────────────────────
    print("\nWarming up GPU (5 steps without profiling) ...")
    data_iter = iter(train_loader)
    for _ in range(5):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        src, tgt, _, _ = [b.to(device) for b in batch]
        dec_in, dec_label = tgt[:, :-1], tgt[:, 1:]
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            logits = model(src, dec_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), dec_label.reshape(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    print("Warmup done.\n")

    # ═══════════════════════════════════════════════════════════
    # PHASE 1 — PyTorch Profiler (for TensorBoard)
    # ═══════════════════════════════════════════════════════════
    print_separator("PHASE 1: PyTorch Profiler (capturing 6 steps)")

    profiler_logdir = "./profiler_logs"
    import os
    os.makedirs(profiler_logdir, exist_ok=True)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=2, warmup=2, active=6, repeat=1),
        on_trace_ready=tensorboard_trace_handler(profiler_logdir),
        with_stack=False,
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for step in range(12):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            src, tgt, _, _ = [b.to(device) for b in batch]
            dec_in, dec_label = tgt[:, :-1], tgt[:, 1:]

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits = model(src, dec_in)
                loss = criterion(logits.reshape(-1, logits.size(-1)), dec_label.reshape(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            prof.step()

    torch.cuda.synchronize()
    print(f"Trace saved to {profiler_logdir}/")
    print(f"View with: tensorboard --logdir={profiler_logdir}")

    # ═══════════════════════════════════════════════════════════
    # PHASE 2 — Manual op-level timing breakdown
    # ═══════════════════════════════════════════════════════════
    print_separator("PHASE 2: Forward pass op-level timing")

    # Get one batch
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        batch = next(data_iter)
    src, tgt, _, _ = [b.to(device) for b in batch]
    dec_in, dec_label = tgt[:, :-1], tgt[:, 1:]
    B, S_src = src.shape
    S_tgt = dec_in.shape[1]
    print(f"Batch: {B} × src_len={S_src} tgt_len={S_tgt}")

    # Hook to time each module in forward
    hook_times = {}
    hook_mem = {}
    hooks = []

    def make_hook(name):
        def pre_hook(module, inp):
            torch.cuda.synchronize()
            hook_times[name] = time.perf_counter()
        def post_hook(module, inp, out):
            torch.cuda.synchronize()
            hook_times[name] = time.perf_counter() - hook_times[name]
        return pre_hook, post_hook

    # Register hooks on top-level modules
    module_hooks = {
        "src_embed": model.src_embed,
        "tgt_embed": model.tgt_embed,
        "src_pos": model.src_pos,
        "tgt_pos": model.tgt_pos,
        "dropout": model.dropout,
        "transformer_enc": model.transformer.encoder,
    }

    for name, mod in module_hooks.items():
        pre, post = make_hook(name)
        hooks.append(mod.register_forward_pre_hook(pre))
        hooks.append(mod.register_forward_hook(post))

    # Warm once
    with torch.amp.autocast("cuda", dtype=amp_dtype):
        _ = model(src, dec_in)
    torch.cuda.synchronize()

    # Time forward
    hook_times.clear()
    with torch.amp.autocast("cuda", dtype=amp_dtype):
        logits = model(src, dec_in)
    torch.cuda.synchronize()

    # Print forward timing
    print(f"\n{'Module':<30} {'Time (ms)':>10}")
    print("-" * 42)
    for name, t in sorted(hook_times.items(), key=lambda x: -x[1]):
        print(f"{name:<30} {t*1000:>10.3f}")
    print("-" * 42)

    # Remove hooks
    for h in hooks:
        h.remove()

    # ═══════════════════════════════════════════════════════════
    # PHASE 3 — FWD vs BWD time ratio
    # ═══════════════════════════════════════════════════════════
    print_separator("PHASE 3: Forward vs Backward vs Optimizer timing")

    num_steps = 10
    fwd_times, bwd_times, opt_times = [], [], []
    data_times = []

    for step_idx in range(num_steps + 3):
        t0 = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        torch.cuda.synchronize()
        t_data = time.perf_counter()

        src, tgt, _, _ = [b.to(device) for b in batch]
        dec_in, dec_label = tgt[:, :-1], tgt[:, 1:]

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            logits = model(src, dec_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), dec_label.reshape(-1))

        torch.cuda.synchronize()
        t2 = time.perf_counter()

        scaler.scale(loss).backward()

        torch.cuda.synchronize()
        t3 = time.perf_counter()

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        t4 = time.perf_counter()

        if step_idx >= 3:  # skip first 3 for warm
            data_times.append(t_data - t0)
            fwd_times.append(t2 - t1)
            bwd_times.append(t3 - t2)
            opt_times.append(t4 - t3)

    avg_data = sum(data_times) / len(data_times) * 1000
    avg_fwd = sum(fwd_times) / len(fwd_times) * 1000
    avg_bwd = sum(bwd_times) / len(bwd_times) * 1000
    avg_opt = sum(opt_times) / len(opt_times) * 1000
    total = avg_fwd + avg_bwd + avg_opt

    print(f"\n{'Phase':<20} {'Time (ms)':>10} {'%':>8}")
    print("-" * 40)
    print(f"{'Data loading':<20} {avg_data:>10.2f} {'-':>8}")
    print(f"{'Forward pass':<20} {avg_fwd:>10.2f} {avg_fwd/total*100:>7.1f}%")
    print(f"{'Backward pass':<20} {avg_bwd:>10.2f} {avg_bwd/total*100:>7.1f}%")
    print(f"{'Optimizer step':<20} {avg_opt:>10.2f} {avg_opt/total*100:>7.1f}%")
    print("-" * 40)
    print(f"{'Fwd+Bwd+Opt':<20} {total:>10.2f}")
    print(f"\n  BWD/FWD ratio: {avg_bwd/avg_fwd:.2f}x (理论 ~2x 为纯 compute, 更大则有 bandwidth 瓶颈)")

    # ═══════════════════════════════════════════════════════════
    # PHASE 4 — GPU memory breakdown
    # ═══════════════════════════════════════════════════════════
    print_separator("PHASE 4: GPU Memory")

    mem = torch.cuda.memory_stats(device)
    print(f"  Peak allocated:     {mem.get('allocated_bytes.all.peak', 0)/1e9:.2f} GB")
    print(f"  Peak reserved:      {mem.get('reserved_bytes.all.peak', 0)/1e9:.2f} GB")
    print(f"  Current allocated:  {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"  Current reserved:   {torch.cuda.memory_reserved()/1e9:.2f} GB")

    # ═══════════════════════════════════════════════════════════
    # PHASE 5 — CUDA kernel statistics
    # ═══════════════════════════════════════════════════════════
    print_separator("PHASE 5: Operator-level CUDA time (via torch.autograd.profiler)")

    # Warm
    for _ in range(2):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        src, tgt, _, _ = [b.to(device) for b in batch]
        dec_in, dec_label = tgt[:, :-1], tgt[:, 1:]
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            logits = model(src, dec_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), dec_label.reshape(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    # Profiled run
    with torch.autograd.profiler.profile(use_cuda=True) as autograd_prof:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        src, tgt, _, _ = [b.to(device) for b in batch]
        dec_in, dec_label = tgt[:, :-1], tgt[:, 1:]
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            logits = model(src, dec_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), dec_label.reshape(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    # Aggregate by operator type
    op_totals = {}
    for evt in autograd_prof.function_events:
        kind = evt.name.split("::")[-1]  # e.g. "aten::mm" -> "mm"
        op_totals[kind] = op_totals.get(kind, 0) + evt.cuda_time_total

    total_cuda = sum(v for k, v in op_totals.items() if k != "cudaLaunchKernel")
    print(f"\n{'Operator':<35} {'CUDA time (ms)':>14} {'%':>8}")
    print("-" * 60)
    for name, t in sorted(op_totals.items(), key=lambda x: -x[1])[:20]:
        pct = t / total_cuda * 100 if total_cuda > 0 else 0
        print(f"{name:<35} {t/1000:>14.3f} {pct:>7.1f}%")
    print("-" * 60)

    # Also aggregate by high-level category
    print()
    print("── 分类汇总 ──")
    categories = {
        "matmul (mm/addmm/scaled_dot_product)": ["mm", "addmm", "scaled_dot_product_attention"],
        "softmax": ["softmax", "_softmax"],
        "layer_norm": ["layer_norm", "native_layer_norm"],
        "activation (relu/gelu/dropout)": ["relu", "gelu", "dropout", "native_dropout"],
        "element-wise (add/mul/copy)": ["add", "mul", "div", "copy_", "clone"],
        "embedding": ["embedding"],
    }
    print(f"{'Category':<40} {'Total ms':>10} {'%':>8}")
    print("-" * 60)
    for cat_name, keys in categories.items():
        cat_ms = sum(op_totals.get(k, 0) for k in keys) / 1000
        pct = cat_ms / (total_cuda/1000) * 100 if total_cuda > 0 else 0
        print(f"{cat_name:<40} {cat_ms:>10.2f} {pct:>7.1f}%")
    other_ms = (total_cuda - sum(
        sum(op_totals.get(k, 0) for k in keys) for keys in categories.values()
    )) / 1000
    print(f"{'其他':<40} {other_ms:>10.2f} {other_ms/(total_cuda/1000)*100:>7.1f}%")

    print()
    print("✓ Profiling complete.")
    print(f"  查看完整时间线: tensorboard --logdir={profiler_logdir}")


if __name__ == "__main__":
    profile_run()
