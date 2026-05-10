"""
Self-Critical Sequence Training (SCST) for NMT.

REINFORCE with a self-critical baseline:
  - Sample a translation → compute BLEU(sample, ref)
  - Greedy decode          → compute BLEU(greedy, ref) as baseline
  - advantage = BLEU_sample - BLEU_greedy
  - loss = -advantage × log P(sampled_sequence | source)

Different sampling strategies are cycled per epoch:
  epoch 1: temperature=0.8
  epoch 2: top-k=50
  epoch 3: multinomial (temperature=1.0)

Usage:
    python rl/rl_train.py                                    # defaults, 3 epochs
    python rl/rl_train.py --epochs 5 --lr 5e-6               # custom settings
    python rl/rl_train.py --strategy temp --temperature 0.6  # single strategy
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

# ── project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config
from model import TranslationTransformer
from tokenizer import load_spm, PAD_ID, BOS_ID, EOS_ID

# ══════════════════════════════════════════════════════════════════════
#  BLEU
# ══════════════════════════════════════════════════════════════════════

def _strip_special(ids: list[int]) -> list[int]:
    return [t for t in ids if t not in (PAD_ID, BOS_ID, EOS_ID)]


def sentence_bleu(hyp_ids: list[int], ref_ids: list[int]) -> float:
    """Sentence-level BLEU-4 with +1 smoothing, computed on token IDs."""
    hyp = _strip_special(hyp_ids)
    ref = _strip_special(ref_ids)

    if len(hyp) == 0:
        return 0.0

    precisions = []
    for n in range(1, 5):
        hyp_ngrams = Counter(tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1))
        ref_ngrams = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
        match = sum(min(hyp_ngrams[g], ref_ngrams.get(g, 0)) for g in hyp_ngrams)
        total = max(len(hyp) - n + 1, 1)
        # +1 smoothing
        match += 1
        total += 1
        precisions.append(match / total)

    # brevity penalty
    if len(hyp) >= len(ref):
        bp = 1.0
    else:
        bp = math.exp(1.0 - len(ref) / max(len(hyp), 1))

    if min(precisions) > 0:
        bleu = bp * math.exp(sum(math.log(p) for p in precisions) / 4.0)
    else:
        bleu = 0.0

    return bleu


def corpus_bleu(hyp_list: list[list[int]], ref_list: list[list[int]]) -> float:
    """Corpus-level BLEU-4 with +1 smoothing."""
    precisions = []
    for n in range(1, 5):
        match_total = 0
        total_total = 0
        for hyp_ids, ref_ids in zip(hyp_list, ref_list):
            hyp = _strip_special(hyp_ids)
            ref = _strip_special(ref_ids)
            hyp_ngrams = Counter(tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1))
            ref_ngrams = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
            match_total += sum(min(hyp_ngrams[g], ref_ngrams.get(g, 0)) for g in hyp_ngrams)
            total_total += max(len(hyp) - n + 1, 0)
        match_total += 1
        total_total += 1
        precisions.append(match_total / max(total_total, 1))

    bp = 1.0
    hyp_len = sum(len(_strip_special(h)) for h in hyp_list)
    ref_len = sum(len(_strip_special(r)) for r in ref_list)
    if hyp_len < ref_len:
        bp = math.exp(1.0 - ref_len / max(hyp_len, 1))

    if min(precisions) > 0:
        return bp * math.exp(sum(math.log(p) for p in precisions) / 4.0)
    return 0.0


# ══════════════════════════════════════════════════════════════════════
#  Decoding strategies
# ══════════════════════════════════════════════════════════════════════

def _encode(model, src_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode source, return (memory, src_pad_mask)."""
    src_pad_mask = (src_ids == PAD_ID)
    src_emb = model.dropout(model.src_pos(model.src_embed(src_ids) * model.embed_scale))
    memory = model.transformer.encoder(src=src_emb, src_key_padding_mask=src_pad_mask)
    return memory, src_pad_mask


@torch.no_grad()
def _decode_step(
    model, memory, src_pad_mask, ys, logits_out, finished,
    temperature: float, top_k: int, top_p: float, bos_id: int, eos_id: int, pad_id: int,
):
    """One autoregressive step.  Modifies logits_out in-place, returns (next_token, finished)."""
    device = ys.device
    tgt_emb = model.dropout(model.tgt_pos(model.tgt_embed(ys) * model.embed_scale))
    tgt_causal_mask = model._create_causal_mask(ys.size(1), device)

    out = model.transformer.decoder(
        tgt=tgt_emb, memory=memory,
        tgt_mask=tgt_causal_mask,
        memory_key_padding_mask=src_pad_mask,
    )
    logits = model.output_proj(out[:, -1, :])  # (B, V)

    # ── temperature ──
    if temperature != 1.0:
        logits = logits / temperature

    # ── top-k ──
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        topk_vals, _ = torch.topk(logits, k, dim=-1)
        threshold = topk_vals[:, -1].unsqueeze(-1)
        logits[logits < threshold] = -float("inf")

    # ── top-p (nucleus) ──
    if top_p > 0.0 and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # shift mask right: keep the first token that pushes cumsum > top_p
        to_remove = cum_probs > top_p
        to_remove[:, 1:] = to_remove[:, :-1].clone()
        to_remove[:, 0] = False
        idx_to_remove = to_remove.scatter(-1, sorted_idx, to_remove)
        logits[idx_to_remove] = -float("inf")

    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

    next_token[finished] = pad_id
    just_finished = (next_token.squeeze(-1) == eos_id)
    finished = finished | just_finished
    logits_out.append(logits)
    return next_token, finished


@torch.no_grad()
def sample_decode(
    model, src_ids: torch.Tensor,
    bos_id: int, eos_id: int, pad_id: int,
    max_len: int = 96,
    temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0,
) -> torch.Tensor:
    """Autoregressive sampling.  Returns token ids (B, S)."""
    B = src_ids.size(0)
    device = src_ids.device
    memory, src_pad_mask = _encode(model, src_ids)

    ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    logits_history = []  # not used during sampling, but kept for symmetry

    for _ in range(max_len):
        next_token, finished = _decode_step(
            model, memory, src_pad_mask, ys, logits_history, finished,
            temperature, top_k, top_p, bos_id, eos_id, pad_id,
        )
        ys = torch.cat([ys, next_token], dim=-1)
        if finished.all():
            break

    return ys


@torch.no_grad()
def greedy_decode(
    model, src_ids: torch.Tensor,
    bos_id: int, eos_id: int, pad_id: int,
    max_len: int = 96,
) -> torch.Tensor:
    """Greedy argmax decode.  Returns token ids (B, S)."""
    B = src_ids.size(0)
    device = src_ids.device
    memory, src_pad_mask = _encode(model, src_ids)

    ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_len):
        tgt_emb = model.dropout(model.tgt_pos(model.tgt_embed(ys) * model.embed_scale))
        tgt_causal_mask = model._create_causal_mask(ys.size(1), device)
        out = model.transformer.decoder(
            tgt=tgt_emb, memory=memory,
            tgt_mask=tgt_causal_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        logits = model.output_proj(out[:, -1, :])
        next_token = logits.argmax(dim=-1, keepdim=True)  # (B, 1)

        next_token[finished] = pad_id
        ys = torch.cat([ys, next_token], dim=-1)

        just_finished = (next_token.squeeze(-1) == eos_id)
        finished = finished | just_finished
        if finished.all():
            break

    return ys


# ══════════════════════════════════════════════════════════════════════
#  SCST training
# ══════════════════════════════════════════════════════════════════════

def collate_rl(
    items: list[tuple[list[int], list[int]]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad src and ref to max length in batch (aligned to 8)."""
    ALIGN = 8
    src_max = max(len(s) for s, _ in items)
    tgt_max = max(len(r) for _, r in items)
    src_max = ((src_max + ALIGN - 1) // ALIGN) * ALIGN
    tgt_max = ((tgt_max + ALIGN - 1) // ALIGN) * ALIGN

    B = len(items)
    src = torch.full((B, src_max), PAD_ID, dtype=torch.long)
    ref = torch.full((B, tgt_max), PAD_ID, dtype=torch.long)
    for i, (s, r) in enumerate(items):
        src[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        ref[i, :len(r)] = torch.tensor(r, dtype=torch.long)
    return src, ref


def rl_train(
    config: Config,
    model: TranslationTransformer,
    data: list[dict],
    sp,
    device: torch.device,
    *,
    epochs: int = 3,
    lr: float = 1e-5,
    batch_size: int = 16,
    sampling_strategies: list[dict] | None = None,
    checkpoint_dir: str = "rl/checkpoints",
    log_interval: int = 5,
):
    """
    SCST fine-tuning.

    sampling_strategies: one dict per epoch, e.g.
        [{"temperature": 0.8}, {"top_k": 50}, {"temperature": 1.0}]
    """
    if sampling_strategies is None:
        sampling_strategies = [
            {"temperature": 0.8, "top_k": 0, "top_p": 0.0},
            {"temperature": 1.0, "top_k": 50, "top_p": 0.0},
            {"temperature": 1.0, "top_k": 0, "top_p": 0.0},
        ]

    # Ensure we have a strategy per epoch
    if len(sampling_strategies) < epochs:
        last = sampling_strategies[-1]
        sampling_strategies.extend([last] * (epochs - len(sampling_strategies)))

    model = model.to(device)
    model.train()

    # Tokenize
    pairs = []
    for item in data:
        s = [BOS_ID] + sp.encode(item["zh"], out_type=int)[:config.max_seq_len - 2] + [EOS_ID]
        r = [BOS_ID] + sp.encode(item["en"], out_type=int)[:config.max_seq_len - 2] + [EOS_ID]
        pairs.append((s, r))

    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_steps = 0
    best_bleu = 0.0

    for epoch in range(1, epochs + 1):
        strat = sampling_strategies[epoch - 1]
        temp = strat.get("temperature", 1.0)
        top_k = strat.get("top_k", 0)
        top_p = strat.get("top_p", 0.0)

        if top_k > 0:
            desc = f"top_k={top_k}"
        elif top_p > 0:
            desc = f"top_p={top_p}"
        else:
            desc = f"temp={temp}"

        print(f"\n{'='*55}")
        print(f"  Epoch {epoch}/{epochs}  |  sampling: {desc}  |  lr={lr:.1e}")
        print(f"{'='*55}")

        # Shuffle
        import random
        random.shuffle(pairs)

        epoch_loss = 0.0
        epoch_bleu_sample = 0.0
        epoch_bleu_greedy = 0.0
        epoch_advantage_pos = 0
        epoch_advantage_total = 0
        n_batches = 0

        for b_start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[b_start:b_start + batch_size]
            src, ref = collate_rl(batch_pairs)
            src, ref = src.to(device), ref.to(device)
            B = src.size(0)

            # ── 1. sample a translation (no grad) ──
            y_sample = sample_decode(
                model, src, BOS_ID, EOS_ID, PAD_ID,
                max_len=config.max_seq_len, temperature=temp,
                top_k=top_k, top_p=top_p,
            )

            # ── 2. greedy baseline (no grad) ──
            y_greedy = greedy_decode(
                model, src, BOS_ID, EOS_ID, PAD_ID,
                max_len=config.max_seq_len,
            )

            # ── 3. BLEU rewards ──
            bleu_sample = torch.tensor(
                [sentence_bleu(y_sample[i].tolist(), ref[i].tolist()) for i in range(B)],
                device=device,
            )
            bleu_greedy = torch.tensor(
                [sentence_bleu(y_greedy[i].tolist(), ref[i].tolist()) for i in range(B)],
                device=device,
            )
            advantage = bleu_sample - bleu_greedy  # (B,)

            # ── 4. log-probabilities (WITH grad) ──
            # Teacher-forcing forward on the sampled sequence.
            # logits[:, t, :] predicts token at position t+1.
            logits = model(src, y_sample)  # (B, S_sample, V)
            log_probs = F.log_softmax(logits, dim=-1)

            # shift: use log_probs[:, :-1, :] → predict y_sample[:, 1:]
            shift_log_probs = log_probs[:, :-1, :]
            shift_targets = y_sample[:, 1:].unsqueeze(-1)
            token_lp = shift_log_probs.gather(-1, shift_targets).squeeze(-1)  # (B, S-1)

            mask = (y_sample[:, 1:] != PAD_ID).float()
            seq_log_prob = (token_lp * mask).sum(dim=-1)  # (B,)

            # ── 5. REINFORCE loss ──
            loss = -(advantage * seq_log_prob).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_steps += 1
            n_batches += 1
            epoch_loss += loss.item()

            with torch.no_grad():
                epoch_bleu_sample += bleu_sample.mean().item()
                epoch_bleu_greedy += bleu_greedy.mean().item()
                epoch_advantage_pos += (advantage > 0).sum().item()
                epoch_advantage_total += B

            if n_batches % log_interval == 0 or b_start + batch_size >= len(pairs):
                avg_loss = epoch_loss / n_batches
                avg_bs = epoch_bleu_sample / n_batches
                avg_bg = epoch_bleu_greedy / n_batches
                adv_pct = epoch_advantage_pos / max(epoch_advantage_total, 1) * 100
                print(
                    f"  batch {n_batches:3d}/{len(pairs)//batch_size + 1:3d} | "
                    f"loss {avg_loss:.4f} | "
                    f"BLEU(sample) {avg_bs:.4f} | "
                    f"BLEU(greedy) {avg_bg:.4f} | "
                    f"adv>0: {adv_pct:.0f}%"
                )

        # ── end of epoch ──
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_bs = epoch_bleu_sample / max(n_batches, 1)
        avg_bg = epoch_bleu_greedy / max(n_batches, 1)

        print(f"\n── Epoch {epoch} summary ──")
        print(f"  loss: {avg_loss:.4f}  |  BLEU sample: {avg_bs:.4f}  |  BLEU greedy: {avg_bg:.4f}")

        # Save checkpoint
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "total_steps": total_steps,
            "bleu_sample": avg_bs,
            "bleu_greedy": avg_bg,
        }
        torch.save(ckpt, out_dir / f"rl_epoch_{epoch}.pt")
        if avg_bs > best_bleu:
            best_bleu = avg_bs
            torch.save(ckpt, out_dir / "rl_best.pt")
            print(f"  → saved best checkpoint (BLEU={best_bleu:.4f})")

    print(f"\nDone.  Best BLEU: {best_bleu:.4f}")
    return best_bleu


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SCST RL fine-tuning for NMT")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt",
                        help="Pretrained model checkpoint")
    parser.add_argument("--data", type=str, default="rl/rl_data.json",
                        help="RL training data (JSON)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of RL fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--strategy", type=str, default="cycle",
                        choices=["cycle", "temp", "topk", "topp", "multinomial"],
                        help="Sampling strategy (cycle rotates per epoch)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Temperature (for temp strategy)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k (for topk strategy)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p (for topp strategy)")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = Config()
    sp = load_spm(config.sp_model_prefix)
    config.vocab_size = sp.get_piece_size()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = TranslationTransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len,
        activation=config.activation,
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    print(f"  loaded (step {ckpt.get('step', '?')}, best_loss {ckpt.get('best_loss', '?'):.4f})")

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
        print("  torch.compile: enabled")

    # Load RL data
    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"RL data: {len(data)} pairs from {args.data}")

    # Build sampling strategies
    if args.strategy == "cycle":
        strategies = [
            {"temperature": 0.8, "top_k": 0, "top_p": 0.0},
            {"temperature": 1.0, "top_k": 50, "top_p": 0.0},
            {"temperature": 1.0, "top_k": 0, "top_p": 0.0},
        ]
    elif args.strategy == "temp":
        strategies = [{"temperature": args.temperature, "top_k": 0, "top_p": 0.0}]
    elif args.strategy == "topk":
        strategies = [{"temperature": 1.0, "top_k": args.top_k, "top_p": 0.0}]
    elif args.strategy == "topp":
        strategies = [{"temperature": 1.0, "top_k": 0, "top_p": args.top_p}]
    elif args.strategy == "multinomial":
        strategies = [{"temperature": 1.0, "top_k": 0, "top_p": 0.0}]
    else:
        strategies = [{"temperature": 0.8, "top_k": 0, "top_p": 0.0}]

    rl_train(
        config, model, data, sp, device,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        sampling_strategies=strategies,
    )


if __name__ == "__main__":
    main()
