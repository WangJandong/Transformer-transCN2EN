#!/usr/bin/env python
"""
Train a Transformer model for Chinese→English translation.

Quick start:
    python train.py                          # train with defaults (pre-tokenized data)
    python train.py --epochs 20              # full training
    python train.py --no_compile --epochs 1  # quick test

Resumes from the latest checkpoint automatically.
"""
import argparse
import datetime
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

from config import Config
from tokenizer import build_tokenizer_if_needed, load_spm
from dataset_tokenized import build_tokenized_dataloaders
from model import TranslationTransformer
from trainer import train, configure_gpu_backend


class Tee:
    """Duplicate stdout to a file (like `| tee log.txt`)."""
    def __init__(self, path: str):
        self.file = open(path, "a", encoding="utf-8", buffering=1)
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        sys.stdout = self.stdout
        self.file.close()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train Chinese→English NMT")
    # All defaults come from config.py — only override if explicitly passed
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--num_encoder_layers", type=int, default=None)
    parser.add_argument("--num_decoder_layers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--compile_mode", type=str, default=None)
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()

    config = Config()
    # Apply CLI overrides (only non-None values)
    for k, v in vars(args).items():
        if v is not None:
            setattr(config, k, v)

    if args.no_compile:
        config.compile_mode = ""
    if args.no_amp:
        config.mixed_precision = False

    # Setup log file (tee to terminal + file)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    tee = Tee(str(log_path))
    print(f"Logging to {log_path}")

    if config.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        config.device = "cpu"

    device = torch.device(config.device)
    configure_gpu_backend()
    set_seed(config.seed)

    # 1. Tokenizer
    sp = build_tokenizer_if_needed(config)
    config.vocab_size = sp.get_piece_size()
    print(f"Vocabulary size: {config.vocab_size}")

    # 2. Data — use pre-tokenized mmap arrays (fast)
    print("Loading pre-tokenized data …", flush=True)
    train_loader, val_loader, total_pairs = build_tokenized_dataloaders(config)
    config._steps_per_epoch = total_pairs // config.batch_size
    print(f"Data ready: {total_pairs:,} training pairs → ~{config._steps_per_epoch:,} steps/epoch")

    # 3. Model
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
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.1f}M")

    # 4. Train
    best_loss = train(config, model, train_loader, val_loader, device)
    print(f"Training done. Best val loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
