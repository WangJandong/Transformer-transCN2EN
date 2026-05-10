"""
Parallel corpus dataset for NMT.
"""
from __future__ import annotations
from typing import Iterator

import numpy as np
import sentencepiece as spm
import torch
from torch.utils.data import IterableDataset, DataLoader


class ParallelIterableDataset(IterableDataset):
    """Stream parallel sentences line-by-line — avoids loading the full corpus into RAM."""

    def __init__(
        self,
        src_path: str,
        tgt_path: str,
        sp: spm.SentencePieceProcessor,
        max_len: int = 256,
        shuffle_buffer: int = 20000,
        seed: int = 42,
        start_line: int = 0,
        end_line: int | None = None,
    ):
        super().__init__()
        self.src_path = src_path
        self.tgt_path = tgt_path
        self.sp = sp
        self.max_len = max_len
        self.shuffle_buffer = shuffle_buffer
        self.rng = np.random.default_rng(seed)
        self.start_line = start_line
        self.end_line = end_line

    def _line_iterator(self) -> Iterator[tuple[str, str]]:
        with open(self.src_path, encoding="utf-8") as fs, open(self.tgt_path, encoding="utf-8") as ft:
            for i, (s, t) in enumerate(zip(fs, ft)):
                if i < self.start_line:
                    continue
                if self.end_line is not None and i >= self.end_line:
                    break
                yield s.strip(), t.strip()

    def _encode(self, text: str) -> list[int]:
        ids = self.sp.encode(text, out_type=int)
        ids = ids[: self.max_len - 2]
        return [BOS_ID, *ids, EOS_ID]

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        buffer: list[tuple[torch.Tensor, torch.Tensor]] = []
        for src, tgt in self._line_iterator():
            src_ids = self._encode(src)
            tgt_ids = self._encode(tgt)
            if len(src_ids) < 4 or len(tgt_ids) < 4:  # skip degenerate pairs
                continue
            item = (
                torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long),
            )
            if len(buffer) < self.shuffle_buffer:
                buffer.append(item)
            else:
                idx = self.rng.integers(0, len(buffer))
                yield buffer[idx]
                buffer[idx] = item
        # drain buffer
        self.rng.shuffle(buffer)
        yield from buffer


BOS_ID = 1
EOS_ID = 2
PAD_ID = 0


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad src and tgt to the max length in the batch, rounded to a multiple of 8.

    Rounding stabilises tensor shapes for torch.compile — without it the compiler
    re-compiles for every unique sequence length and hits the recompile limit (8).
    """
    src_list, tgt_list = zip(*batch)

    src_max = max(t.size(0) for t in src_list)
    tgt_max = max(t.size(0) for t in tgt_list)

    # Round up to nearest multiple of 8 to limit unique shapes for torch.compile
    ALIGN = 8
    src_max = ((src_max + ALIGN - 1) // ALIGN) * ALIGN
    tgt_max = ((tgt_max + ALIGN - 1) // ALIGN) * ALIGN

    def pad_to(seqs: list[torch.Tensor], length: int) -> torch.Tensor:
        out = torch.full((len(seqs), length), PAD_ID, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :s.size(0)] = s
        return out

    src_padded = pad_to(src_list, src_max)
    tgt_padded = pad_to(tgt_list, tgt_max)

    src_mask = (src_padded != PAD_ID)
    tgt_mask = (tgt_padded != PAD_ID)
    return src_padded, tgt_padded, src_mask, tgt_mask


def build_dataloaders(config, sp) -> tuple[DataLoader, DataLoader]:
    train_dataset = ParallelIterableDataset(
        src_path=str(config.data_dir / f"train.{config.src_lang}"),
        tgt_path=str(config.data_dir / f"train.{config.tgt_lang}"),
        sp=sp,
        max_len=config.max_seq_len,
        shuffle_buffer=20000,
        seed=config.seed,
        end_line=config.max_train_samples if config.max_train_samples > 0 else None,
    )

    val_dataset = ParallelIterableDataset(
        src_path=str(config.data_dir / f"val.{config.src_lang}"),
        tgt_path=str(config.data_dir / f"val.{config.tgt_lang}"),
        sp=sp,
        max_len=config.max_seq_len,
        shuffle_buffer=2000,
        seed=config.seed,
    )

    dl_kwargs = dict(
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    if config.num_workers > 0:
        dl_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, **dl_kwargs)
    val_loader = DataLoader(val_dataset, **dl_kwargs)

    return train_loader, val_loader
