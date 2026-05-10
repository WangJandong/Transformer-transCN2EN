"""Show example translations on the 363 RL data pairs."""
import json, sys
sys.path.insert(0, ".")
from config import Config
from model import TranslationTransformer
from tokenizer import load_spm, BOS_ID, EOS_ID
import torch

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

with open("rl/rl_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

examples = [0, 2, 5, 42, 78, 120, 165, 200, 250, 300, 350]

for idx in examples:
    if idx >= len(data):
        continue
    item = data[idx]
    ids = [BOS_ID] + sp.encode(item["zh"], out_type=int)[:config.max_seq_len - 2] + [EOS_ID]
    src = torch.tensor([ids], dtype=torch.long, device=device)
    out_ids = model.translate(src, BOS_ID, EOS_ID, max_len=config.max_seq_len, beam_size=1)
    out_ids = out_ids[0].tolist()
    out_ids = [t for t in out_ids if t not in (BOS_ID, EOS_ID, 0)]
    hyp = sp.decode(out_ids)
    print(f"[{idx}] SRC: {item['zh']}")
    print(f"    REF: {item['en']}")
    print(f"    HYP: {hyp}")
    print()
