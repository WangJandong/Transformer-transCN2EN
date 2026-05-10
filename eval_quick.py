"""Quick offline eval: BLEU + chrF + length breakdown. No network needed."""
import random, time, sys
import torch, sacrebleu
from config import Config
from model import TranslationTransformer
from tokenizer import load_spm, BOS_ID, EOS_ID

N = 200
BEAM = 1
config = Config()
sp = load_spm(config.sp_model_prefix)
config.vocab_size = sp.get_piece_size()
device = torch.device("cuda")

print(f"Loading model...")
model = TranslationTransformer(
    vocab_size=config.vocab_size, d_model=config.d_model, nhead=config.nhead,
    num_encoder_layers=config.num_encoder_layers, num_decoder_layers=config.num_decoder_layers,
    dim_feedforward=config.dim_feedforward, dropout=config.dropout,
    max_seq_len=config.max_seq_len, activation=config.activation,
).to(device)
ckpt = torch.load("checkpoints/best.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"Loaded. step={ckpt.get('step','?')}, loss={ckpt.get('best_loss',0):.4f}")
print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

with open("data/test.zh", encoding="utf-8") as f:
    all_src = [l.strip() for l in f if l.strip()]
with open("data/test.en", encoding="utf-8") as f:
    all_ref = [l.strip() for l in f if l.strip()]

random.seed(42)
indices = random.sample(range(len(all_src)), N)
srcs = [all_src[i] for i in indices]
refs = [all_ref[i] for i in indices]

print(f"Translating {N} sentences (beam={BEAM})...")
t0 = time.time()
hyps = []
for i, text in enumerate(srcs):
    ids = [BOS_ID] + sp.encode(text, out_type=int)[:config.max_seq_len - 2] + [EOS_ID]
    src = torch.tensor([ids], dtype=torch.long, device=device)
    out_ids = model.translate(src, BOS_ID, EOS_ID, max_len=config.max_seq_len, beam_size=BEAM)
    out_ids = out_ids[0].tolist()
    out_ids = [t for t in out_ids if t not in (BOS_ID, EOS_ID, 0)]
    hyps.append(sp.decode(out_ids))
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{N}")
print(f"Done in {time.time()-t0:.1f}s ({N/(time.time()-t0):.0f} sent/s)")

print()
print("=" * 60)
print("  EVALUATION RESULTS")
print("=" * 60)
bleu = sacrebleu.corpus_bleu(hyps, [refs])
print(f"  BLEU:  {bleu}")
chrf = sacrebleu.corpus_chrf(hyps, [refs])
print(f"  chrF:  {chrf}")

# Length-stratified
print()
print("  LENGTH BREAKDOWN:")
buckets = [
    ("short  (1-10w)",  1, 10),
    ("medium (11-25w)", 11, 25),
    ("long   (26-50w)", 26, 50),
    ("xlong  (51+w)",   51, 9999),
]
for name, lo, hi in buckets:
    idx = [i for i, r in enumerate(refs) if lo <= len(r.split()) <= hi]
    if len(idx) < 5:
        print(f"  {name}: n={len(idx)} (too few)")
        continue
    b = sacrebleu.corpus_bleu([hyps[i] for i in idx], [[refs[i] for i in idx]])
    c = sacrebleu.corpus_chrf([hyps[i] for i in idx], [[refs[i] for i in idx]])
    print(f"  {name}: n={len(idx):<5} BLEU={b.score:5.1f}  chrF={c.score:5.1f}")

# Examples
print()
print("  SAMPLE OUTPUTS:")
random.seed(123)
for i in random.sample(range(N), 5):
    print(f"  [{i}] SRC: {srcs[i][:120]}")
    print(f"      REF: {refs[i][:120]}")
    print(f"      HYP: {hyps[i][:120]}")
    print()
