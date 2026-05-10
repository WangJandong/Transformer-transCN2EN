"""
Interactive translation with a trained checkpoint.
Usage:
    python translate.py --text "今天天气很好。" --checkpoint checkpoints/best.pt
    python translate.py --file input.zh --checkpoint checkpoints/best.pt
"""
import argparse
import sys
from pathlib import Path

import torch

from config import Config
from model import TranslationTransformer
from tokenizer import load_spm, BOS_ID, EOS_ID


@torch.no_grad()
def translate_text(
    text: str,
    model: TranslationTransformer,
    sp,
    config: Config,
    device: torch.device,
    beam_size: int = 1,
) -> str:
    ids = sp.encode(text, out_type=int)[: config.max_seq_len - 2]
    ids = [BOS_ID, *ids, EOS_ID]
    src = torch.tensor([ids], dtype=torch.long, device=device)

    model.eval()
    out_ids = model.translate(src, BOS_ID, EOS_ID, max_len=config.max_seq_len, beam_size=beam_size)
    out_ids = out_ids[0].tolist()

    # strip special tokens
    out_ids = [t for t in out_ids if t not in (BOS_ID, EOS_ID, 0)]
    return sp.decode(out_ids)


def main():
    parser = argparse.ArgumentParser(description="Chinese → English Translator")
    parser.add_argument("--text", type=str, help="Single sentence in Chinese")
    parser.add_argument("--file", type=str, help="File with one Chinese sentence per line")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--beam", type=int, default=4, help="Beam size (default: 4)")
    parser.add_argument("--spm_prefix", type=str, default="spm_bpe")
    args = parser.parse_args()

    config = Config()
    sp = load_spm(args.spm_prefix)
    config.vocab_size = sp.get_piece_size()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from step {ckpt.get('step', '?')}")

    if args.text:
        result = translate_text(args.text, model, sp, config, device, args.beam)
        print(result)
    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        for line in lines:
            result = translate_text(line, model, sp, config, device, args.beam)
            print(result)
    else:
        print("Enter Chinese sentences (Ctrl+C to quit):")
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                result = translate_text(line, model, sp, config, device, args.beam)
                print(f"> {result}\n")
        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    main()
