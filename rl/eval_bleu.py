"""Evaluate model BLEU on test set using sacreBLEU."""
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config
from model import TranslationTransformer
from tokenizer import load_spm, BOS_ID, EOS_ID

import torch

# Config
CHECKPOINT = "checkpoints/best.pt"
TEST_SRC = "data/test.zh"
TEST_REF = "data/test.en"
N_SAMPLES = 2000
DEVICE = "cuda"
MAX_LEN = 96

print(f"Loading model from {CHECKPOINT} ...")
config = Config()
sp = load_spm(config.sp_model_prefix)
config.vocab_size = sp.get_piece_size()
device = torch.device(DEVICE)

model = TranslationTransformer(
    vocab_size=config.vocab_size,
    d_model=config.d_model, nhead=config.nhead,
    num_encoder_layers=config.num_encoder_layers,
    num_decoder_layers=config.num_decoder_layers,
    dim_feedforward=config.dim_feedforward,
    dropout=config.dropout, max_seq_len=config.max_seq_len,
    activation=config.activation,
).to(device)

ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"  Loaded step {ckpt.get('step', '?')}, best_loss={ckpt.get('best_loss', '?'):.4f}")

# Load test data
print(f"Loading test data, sampling {N_SAMPLES} ...")
with open(TEST_SRC, encoding="utf-8") as f:
    src_lines = [l.strip() for l in f if l.strip()]
with open(TEST_REF, encoding="utf-8") as f:
    ref_lines = [l.strip() for l in f if l.strip()]

# Sample
random.seed(42)
indices = random.sample(range(min(len(src_lines), len(ref_lines))), N_SAMPLES)

# Translate
print("Translating ...")
hyps = []
refs = []
for i, idx in enumerate(indices):
    src = src_lines[idx]
    ref = ref_lines[idx]

    ids = [BOS_ID] + sp.encode(src, out_type=int)[:MAX_LEN - 2] + [EOS_ID]
    src_tensor = torch.tensor([ids], dtype=torch.long, device=device)

    out_ids = model.translate(src_tensor, BOS_ID, EOS_ID, max_len=MAX_LEN, beam_size=1)
    out_ids = out_ids[0].tolist()
    out_ids = [t for t in out_ids if t not in (BOS_ID, EOS_ID, 0)]
    hyp = sp.decode(out_ids)

    hyps.append(hyp)
    refs.append(ref)

    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{N_SAMPLES} ...")

# Compute BLEU
try:
    import sacrebleu
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    print(f"\n{'='*50}")
    print(f"  sacreBLEU on {N_SAMPLES} test samples:")
    print(f"  {bleu}")
    print(f"{'='*50}")
except ImportError:
    # Fallback: use nltk
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        refs_tok = [r.split() for r in refs]
        hyps_tok = [h.split() for h in hyps]
        smooth = SmoothingFunction().method1
        bleu = corpus_bleu([[r] for r in refs_tok], hyps_tok, smoothing_function=smooth)
        print(f"\n  NLTK BLEU (corpus): {bleu*100:.2f}")
    except ImportError:
        print("\n  Neither sacrebleu nor nltk available. Install one:")
        print("    pip install sacrebleu")
