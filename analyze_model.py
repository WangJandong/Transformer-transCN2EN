"""Print a detailed parameter breakdown of the translation model."""
from collections import defaultdict
from config import Config
from model import TranslationTransformer

config = Config()
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

total = sum(p.numel() for p in model.parameters())

# Manual breakdown from the actual param structure
V = config.vocab_size       # 32000
d = config.d_model          # 512
df = config.dim_feedforward # 2048
L = config.max_seq_len      # 256
n_enc = config.num_encoder_layers
n_dec = config.num_decoder_layers

# --- per-layer counts ---
# Self-attention: QKV merged proj + out_proj
self_attn_params = 3 * d * d + d + d * d + d  # in_proj_w + in_proj_b + out_proj_w + out_proj_b
ffn_params = d * df + df + df * d + d          # linear1_w + linear1_b + linear2_w + linear2_b
enc_norm_params = 2 * (2 * d)                   # norm1 + norm2 (weight + bias each)
dec_norm_params = 3 * (2 * d)                   # norm1 + norm2 + norm3

enc_per_layer = self_attn_params + ffn_params + enc_norm_params
dec_per_layer = 2 * self_attn_params + ffn_params + dec_norm_params  # self + cross attn

enc_total = n_enc * enc_per_layer + 2 * d   # + final norm
dec_total = n_dec * dec_per_layer + 2 * d   # + final norm

src_embed = V * d
tgt_embed = V * d
pos_each = L * d

output_proj = V * d + V

embed_total = src_embed + tgt_embed + 2 * pos_each

print(f"{'='*60}")
print(f"  MODEL PARAMETER BREAKDOWN — 93.6M total")
print(f"{'='*60}")
print(f"  Config: d={d}, d_ff={df}, heads={config.nhead}, vocab={V/1000:.0f}K")
print(f"  Encoder layers: {n_enc}  |  Decoder layers: {n_dec}")
print()

# Table
def row(name, count, pct, indent=2):
    bar = "█" * max(0, int(pct * 2.5))
    print(f"{' ' * indent}{name:<28} {count:>10,}  ({pct:5.1f}%)  {bar}")

print("  ── Embeddings ──────────────────────────────────────")
row("src_embed  (32K×512)",   src_embed,   src_embed / total * 100)
row("tgt_embed  (32K×512)",   tgt_embed,   tgt_embed / total * 100)
row("src_pos    (256×512)",   pos_each,    pos_each / total * 100)
row("tgt_pos    (256×512)",   pos_each,    pos_each / total * 100)
print(f"    {'─'*50}")
row("Subtotal",               embed_total, embed_total / total * 100)
print()
print("  ── Encoder (6 layers) ───────────────────────────────")
row("Self-attention ×6",  n_enc * self_attn_params, n_enc * self_attn_params / total * 100)
row("FFN ×6",             n_enc * ffn_params,       n_enc * ffn_params / total * 100)
row("LayerNorm ×6+final", n_enc * enc_norm_params + 2 * d, (n_enc * enc_norm_params + 2 * d) / total * 100)
print(f"    {'─'*50}")
row("Subtotal",           enc_total, enc_total / total * 100)
print()
print("  ── Decoder (6 layers) ───────────────────────────────")
row("Self-attention ×6",  n_dec * self_attn_params, n_dec * self_attn_params / total * 100)
row("Cross-attention ×6", n_dec * self_attn_params, n_dec * self_attn_params / total * 100)
row("FFN ×6",             n_dec * ffn_params,       n_dec * ffn_params / total * 100)
row("LayerNorm ×6+final", n_dec * dec_norm_params + 2 * d, (n_dec * dec_norm_params + 2 * d) / total * 100)
print(f"    {'─'*50}")
row("Subtotal",           dec_total, dec_total / total * 100)
print()
print("  ── Output ───────────────────────────────────────────")
row("output_proj (512×32K)", V * d, V * d / total * 100)
row("output bias",           V,     V / total * 100)
print(f"    {'─'*50}")
row("Subtotal",              output_proj, output_proj / total * 100)
print()
print(f"  {'='*60}")
row("TOTAL", total, 100.0, indent=0)
print()

# Grouped perspective
print(f"  {'─'*60}")
print(f"  ALTERNATIVE VIEW: module families")
print(f"  {'─'*60}")
families = [
    ("Embedding matrices (4×)",   embed_total, total),
    ("Attention (all QKV+out)",   (n_enc * 1 + n_dec * 2) * self_attn_params, total),
    ("Feed-forward (all FFN)",    (n_enc + n_dec) * ffn_params, total),
    ("LayerNorm (all)",           n_enc * enc_norm_params + n_dec * dec_norm_params + 4 * d, total),
    ("Output projection",         output_proj, total),
]
for name, n, t in families:
    print(f"    {name:<32} {n:>10,}  ({n/t*100:5.1f}%)")
print()

# Key insight
print(f"  ⚡ KEY INSIGHT")
print(f"  {'─'*60}")
print(f"  Embeddings (src+tgt) account for {src_embed+tgt_embed:,} params")
print(f"  ({ (src_embed+tgt_embed)/total*100:.1f}% of total).")
print(f"  If src_embed, tgt_embed, and output_proj share weights")
print(f"  (weight tying), params drop from {total/1e6:.1f}M to")
print(f"  {(total - src_embed - tgt_embed)/1e6:.1f}M.")
print(f"  {'='*60}")
