"""Compare model output on WMT test vs custom news data."""
import random, sys
sys.path.insert(0, ".")
from config import Config
from model import TranslationTransformer
from tokenizer import load_spm, BOS_ID, EOS_ID
import torch
import sacrebleu

config = Config()
sp = load_spm(config.sp_model_prefix)
config.vocab_size = sp.get_piece_size()
device = torch.device("cuda")

model = TranslationTransformer(
    vocab_size=config.vocab_size, d_model=config.d_model, nhead=config.nhead,
    num_encoder_layers=config.num_encoder_layers, num_decoder_layers=config.num_decoder_layers,
    dim_feedforward=config.dim_feedforward, dropout=config.dropout,
    max_seq_len=config.max_seq_len, activation=config.activation,
).to(device)

ckpt = torch.load("checkpoints/best.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()

# Load WMT test
with open("data/test.zh", encoding="utf-8") as f:
    wmt_src = [l.strip() for l in f if l.strip()]
with open("data/test.en", encoding="utf-8") as f:
    wmt_ref = [l.strip() for l in f if l.strip()]

def translate(text):
    ids = [BOS_ID] + sp.encode(text, out_type=int)[:config.max_seq_len - 2] + [EOS_ID]
    src = torch.tensor([ids], dtype=torch.long, device=device)
    out_ids = model.translate(src, BOS_ID, EOS_ID, max_len=config.max_seq_len, beam_size=1)
    out_ids = out_ids[0].tolist()
    out_ids = [t for t in out_ids if t not in (BOS_ID, EOS_ID, 0)]
    return sp.decode(out_ids)

# WMT test examples
print("=" * 70)
print("  WMT TEST SET samples (word-segmented Chinese input)")
print("=" * 70)
random.seed(42)
for idx in random.sample(range(len(wmt_src)), 5):
    hyp = translate(wmt_src[idx])
    print(f"  SRC: {wmt_src[idx][:80]}")
    print(f"  REF: {wmt_ref[idx][:80]}")
    print(f"  HYP: {hyp[:80]}")
    print()

# WMT BLEU on a small sample
print("=" * 70)
print("  BLEU comparison (100 random samples each)")
print("=" * 70)
random.seed(123)
indices = random.sample(range(len(wmt_src)), 100)
wmt_hyps = [translate(wmt_src[i]) for i in indices]
wmt_refs = [wmt_ref[i] for i in indices]
bleu_wmt = sacrebleu.corpus_bleu(wmt_hyps, [wmt_refs], force=True)
print(f"  WMT test (100): {bleu_wmt}")

# Custom news BLEU
import json
with open("rl/news_data.json", "r", encoding="utf-8") as f:
    news = json.load(f)
news_hyps = [translate(d["zh"]) for d in news]
news_refs = [d["en"] for d in news]
bleu_news = sacrebleu.corpus_bleu(news_hyps, [news_refs], force=True)
print(f"  Custom news (100): {bleu_news}")
