"""
Training loop with Noam LR schedule, mixed precision, and Turing-optimized kernels.
"""
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from model import TranslationTransformer


def configure_gpu_backend() -> None:
    """Set CUDA flags for Turing SM 7.5 (RTX 2080 Ti) to maximise throughput."""
    if not torch.cuda.is_available():
        return

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

    device = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {device}  |  SM {cap[0]}.{cap[1]}  |  "
          f"VRAM {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print("GPU backend: TF32=off  fp16_reduce=on  cudnn_benchmark=on")


def noam_lr(step: int, d_model: int, warmup_steps: int) -> float:
    step = max(step, 1)
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))


def save_checkpoint(model, optimizer, scheduler, step, best_loss, config, is_best=False):
    config.checkpoint_dir.mkdir(exist_ok=True)
    # torch.compile wraps model; save the underlying module for portability
    unwrapped = model._orig_mod if hasattr(model, "_orig_mod") else model
    ckpt = {
        "model": unwrapped.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "best_loss": best_loss,
    }
    path = config.checkpoint_dir / f"step_{step}.pt"
    torch.save(ckpt, path)
    # Keep only the last 5 checkpoints + best.pt to save disk space
    all_ckpts = sorted(
        [p for p in config.checkpoint_dir.glob("step_*.pt")],
        key=lambda x: int(x.stem.split("_")[1]),
    )
    for old in all_ckpts[:-5]:
        old.unlink()
    if is_best:
        best_path = config.checkpoint_dir / "best.pt"
        torch.save(ckpt, best_path)
    return path


def load_checkpoint(model, optimizer, scheduler, config):
    best_path = config.checkpoint_dir / "best.pt"
    candidates = sorted(
        [p for p in config.checkpoint_dir.glob("step_*.pt") if p != best_path],
        key=lambda x: int(x.stem.split("_")[1]),
    )
    path = candidates[-1] if candidates else (best_path if best_path.exists() else None)
    if path is None:
        return 0, float("inf")
    ckpt = torch.load(path, map_location=config.device, weights_only=False)
    # torch.compile wraps the model in OptimizedModule; keys get _orig_mod. prefix.
    # Strip prefix if checkpoint was saved without it.
    state = ckpt["model"]
    sample_key = next(iter(model.state_dict()))
    if sample_key.startswith("_orig_mod.") and not next(iter(state)).startswith("_orig_mod."):
        state = {"_orig_mod." + k: v for k, v in state.items()}
    elif not sample_key.startswith("_orig_mod.") and next(iter(state)).startswith("_orig_mod."):
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["step"], ckpt["best_loss"]


def train(config, model, train_loader, val_loader, device):
    model = model.to(device)

    # torch.compile — "eager" backend: Dynamo graph capture without inductor codegen.
    # Stable on PyTorch 2.8 + Turing SM 7.5. Gives ~33% speedup.
    if config.compile_mode and hasattr(torch, "compile"):
        backend = "cudagraphs"
        print(f"torch.compile backend={backend} …", flush=True)
        model = torch.compile(model, backend=backend)
        print("torch.compile: done")
    else:
        print("torch.compile: skipped")

    # Fused AdamW
    fused_kwargs = {}
    if config.fused_adamw and torch.cuda.is_available():
        fused_kwargs = {"fused": True}
        print("AdamW: fused CUDA kernel")
    else:
        print("AdamW: eager (no fused)")

    optimizer = AdamW(
        model.parameters(), lr=config.lr,
        betas=(0.9, 0.98), eps=1e-9, **fused_kwargs,
    )
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda s: noam_lr(s, config.d_model, config.warmup_steps),
    )
    label_smoothing = config.label_smoothing if config.label_smoothing > 0 else 0.0

    start_step, best_loss = load_checkpoint(model, optimizer, scheduler, config)
    if start_step > 0:
        print(f"Resumed from step {start_step}, best loss {best_loss:.4f}")

    use_amp = config.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    amp_dtype = torch.float16
    model.train()

    total_tokens = 0
    t0 = time.perf_counter()
    step = start_step
    best_val_loss = best_loss

    for epoch in range(1, config.epochs + 1):
        for batch in train_loader:
            step += 1
            src, tgt, _, _ = [b.to(device, non_blocking=True) for b in batch]

            dec_input = tgt[:, :-1]
            dec_label = tgt[:, 1:]

            # CUDA graph needs step boundary marker
            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()

            # Forward
            amp_ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()
            with amp_ctx:
                logits = model(src, dec_input)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    dec_label.reshape(-1),
                    ignore_index=0,
                    label_smoothing=label_smoothing,
                )
                loss = loss / config.grad_accum_steps

            # Backward
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step
            if step % config.grad_accum_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            total_tokens += src.size(0) * (src.size(1) + tgt.size(1))

            if step % config.log_interval == 0:
                elapsed = time.perf_counter() - t0
                tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0
                pct = step / config._steps_per_epoch * 100 if config._steps_per_epoch > 0 else 0
                steps_since_log = min(step, config.log_interval)  # uses current interval, not total step
                rate = steps_since_log / max(elapsed, 0.001)
                eta_sec = (config._steps_per_epoch - step) / max(rate, 0.001) if config._steps_per_epoch > 0 else 0
                eta_min = eta_sec / 60
                print(
                    f"ep {epoch}/{config.epochs} | {pct:5.1f}% | "
                    f"step {step:7d} | "
                    f"loss {loss.item() * config.grad_accum_steps:.4f} | "
                    f"lr {scheduler.get_last_lr()[0]:.2e} | "
                    f"{tok_per_sec:,.0f} tok/s | "
                    f"ETA {eta_min:.0f}min"
                )
                total_tokens = 0
                t0 = time.perf_counter()

            if step % config.save_interval == 0:
                save_checkpoint(model, optimizer, scheduler, step, best_val_loss, config)

            if step % config.val_interval == 0:
                val_loss = validate(config, model, val_loader, label_smoothing, device)
                model.train()
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                print(f"--- val loss {val_loss:.4f} {'(best)' if is_best else ''} ---")
                save_checkpoint(model, optimizer, scheduler, step, best_val_loss, config, is_best=is_best)

    return best_val_loss


@torch.no_grad()
def validate(config, model, val_loader, label_smoothing, device):
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in val_loader:
        src, tgt, _, _ = [b.to(device) for b in batch]
        dec_input = tgt[:, :-1]
        dec_label = tgt[:, 1:]

        logits = model(src, dec_input)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            dec_label.reshape(-1),
            ignore_index=0,
            label_smoothing=label_smoothing,
        )
        total_loss += loss.item()
        n += 1
        if n >= 200:
            break

    return total_loss / max(n, 1)
