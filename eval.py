"""
Comprehensive translation quality evaluation.
Metrics: BLEU, chrF, BERTScore, COMET + length-stratified breakdown.

Usage:
    python eval.py                          # default: 2000 samples
    python eval.py --samples 5000           # more samples
    python eval.py --no-comet              # skip COMET (faster)
    python eval.py --checkpoint checkpoints/step_415000.pt
"""
import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import sacrebleu

from config import Config
from model import TranslationTransformer
from tokenizer import load_spm, BOS_ID, EOS_ID


def load_model(checkpoint_path: str, config: Config, device: torch.device):
    sp = load_spm(config.sp_model_prefix)
    config.vocab_size = sp.get_piece_size()

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
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, sp, ckpt


@torch.no_grad()
def translate_batch(model, sp, texts, config, device, beam_size):
    hyps = []
    for text in texts:
        ids = [BOS_ID] + sp.encode(text, out_type=int)[:config.max_seq_len - 2] + [EOS_ID]
        src = torch.tensor([ids], dtype=torch.long, device=device)
        out_ids = model.translate(src, BOS_ID, EOS_ID, max_len=config.max_seq_len, beam_size=beam_size)
        out_ids = out_ids[0].tolist()
        out_ids = [t for t in out_ids if t not in (BOS_ID, EOS_ID, 0)]
        hyps.append(sp.decode(out_ids))
    return hyps


def compute_bleu(hyps, refs):
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    return {
        "BLEU": round(bleu.score, 1),
        "BLEU (sig)": str(bleu).split()[2] if bleu.score > 0 else "—",
    }


def compute_chrf(hyps, refs):
    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    return {"chrF": round(chrf.score, 1)}


def compute_bertscore(hyps, refs, device):
    from bert_score import score
    P, R, F1 = score(hyps, refs, lang="en", device=device, verbose=False, batch_size=32)
    return {
        "BERTScore P": round(P.mean().item() * 100, 1),
        "BERTScore R": round(R.mean().item() * 100, 1),
        "BERTScore F1": round(F1.mean().item() * 100, 1),
    }


def compute_comet(hyps, refs, srcs, gpus):
    from comet import load_from_checkpoint, download_model
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    model.eval()

    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
    output = model.predict(data, batch_size=16, gpus=gpus, progress_bar=False)
    return {
        "COMET (seg-avg)": round(output.system_score, 4),
    }


def length_bucket_bleu(hyps, refs):
    buckets = [
        ("short  (1-10w)",  1, 10),
        ("medium (11-25w)", 11, 25),
        ("long   (26-50w)", 26, 50),
        ("xlong  (51+w)",   51, 9999),
    ]
    results = {}
    for name, lo, hi in buckets:
        idx = [i for i, r in enumerate(refs) if lo <= len(r.split()) <= hi]
        if len(idx) < 5:
            results[name] = f"n={len(idx):<5}  too few samples"
            continue
        h = [hyps[i] for i in idx]
        r = [refs[i] for i in idx]
        bleu = sacrebleu.corpus_bleu(h, [r])
        chrf = sacrebleu.corpus_chrf(h, [r])
        results[name] = f"n={len(idx):<5}  BLEU={bleu.score:5.1f}  chrF={chrf.score:5.1f}"
    return results


def show_examples(hyps, refs, srcs, indices):
    lines = []
    for i in indices:
        lines.append(f"  [{i}] SRC: {srcs[i][:100]}")
        lines.append(f"      REF: {refs[i][:100]}")
        lines.append(f"      HYP: {hyps[i][:100]}")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate translation model quality")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-comet", action="store_true")
    parser.add_argument("--no-bertscore", action="store_true")
    args = parser.parse_args()

    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpus = 1 if device.type == "cuda" else 0

    print(f"{'='*65}")
    print(f"  TRANSLATION MODEL EVALUATION")
    print(f"{'='*65}")
    print(f"  Device:     {device}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Samples:    {args.samples}")
    print(f"  Beam:       {args.beam}")
    print()

    # ── Load model ──────────────────────────────────────────────
    t0 = time.time()
    model, sp, ckpt = load_model(args.checkpoint, config, device)
    print(f"  Model loaded (step={ckpt.get('step', '?')}, "
          f"loss={ckpt.get('best_loss', ckpt.get('loss', 0)):.4f}) "
          f"in {time.time() - t0:.1f}s")
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {param_count:.1f}M")
    print()

    # ── Load test data ──────────────────────────────────────────
    with open(config.data_dir / "test.zh", encoding="utf-8") as f:
        all_src = [l.strip() for l in f if l.strip()]
    with open(config.data_dir / "test.en", encoding="utf-8") as f:
        all_ref = [l.strip() for l in f if l.strip()]
    assert len(all_src) == len(all_ref)

    random.seed(args.seed)
    indices = random.sample(range(len(all_src)), args.samples)
    srcs = [all_src[i] for i in indices]
    refs = [all_ref[i] for i in indices]

    print(f"  Test set: {len(all_src):,} pairs total, sampled {args.samples:,}")
    print()

    # ── Translate ───────────────────────────────────────────────
    print(f"  Translating {args.samples:,} sentences (beam={args.beam}) ...")
    t0 = time.time()
    hyps = translate_batch(model, sp, srcs, config, device, args.beam)
    t_trans = time.time() - t0
    print(f"  Done in {t_trans:.1f}s ({args.samples / t_trans:.0f} sent/s)")
    print()

    # ── BLEU & chrF ─────────────────────────────────────────────
    print(f"{'─'*65}")
    print("  LEXICAL METRICS (n-gram overlap)")
    print(f"{'─'*65}")
    for k, v in compute_bleu(hyps, refs).items():
        print(f"  {k:<18} {v}")
    for k, v in compute_chrf(hyps, refs).items():
        print(f"  {k:<18} {v}")
    print()

    # ── BERTScore ───────────────────────────────────────────────
    if not args.no_bertscore:
        print(f"{'─'*65}")
        print("  BERTSCORE (semantic similarity via BERT embeddings)")
        print(f"{'─'*65}")
        t0 = time.time()
        for k, v in compute_bertscore(hyps, refs, device).items():
            print(f"  {k:<18} {v}")
        print(f"  {'time':<18} {time.time() - t0:.1f}s")
        print()

    # ── COMET ───────────────────────────────────────────────────
    if not args.no_comet:
        print(f"{'─'*65}")
        print("  COMET (neural quality estimation, wmt22-comet-da)")
        print(f"{'─'*65}")
        try:
            t0 = time.time()
            for k, v in compute_comet(hyps, refs, srcs, gpus).items():
                print(f"  {k:<18} {v}")
            print(f"  {'time':<18} {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"  COMET failed: {e}")
        print()

    # ── Length-stratified ───────────────────────────────────────
    print(f"{'─'*65}")
    print("  LENGTH-STRATIFIED BREAKDOWN")
    print(f"{'─'*65}")
    for bucket, result in length_bucket_bleu(hyps, refs).items():
        print(f"  {bucket:<18} {result}")
    print()

    # ── Examples ────────────────────────────────────────────────
    print(f"{'─'*65}")
    print("  SAMPLE OUTPUTS")
    print(f"{'─'*65}")
    example_indices = random.sample(range(args.samples), min(6, args.samples))
    print(show_examples(hyps, refs, srcs, example_indices))

    print(f"{'='*65}")
    print("  Evaluation complete.")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
