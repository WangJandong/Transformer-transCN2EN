"""DataLoader for pre-tokenized data — reads int32 numpy arrays directly."""
from __future__ import annotations
from typing import Iterator
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2


class PreTokenizedDataset(IterableDataset):
    """Stream pre-tokenized int32 arrays. No CPU SentencePiece during training."""

    def __init__(self, ids_path: str, offsets_path: str, lengths_path: str,
                 shuffle_buffer: int = 20000, seed: int = 42,
                 start_idx: int = 0, end_idx: int | None = None):
        super().__init__()
        self.all_ids = np.load(ids_path, mmap_mode="r")   # memory-mapped
        self.offsets = np.load(offsets_path, mmap_mode="r")
        self.lengths = np.load(lengths_path, mmap_mode="r")
        self.shuffle_buffer = shuffle_buffer
        self.rng = np.random.default_rng(seed)
        self.start_idx = start_idx
        self.end_idx = end_idx if end_idx is not None else len(self.lengths)

    def __iter__(self) -> Iterator[torch.Tensor]:
        n = min(self.end_idx, len(self.lengths))
        indices = np.arange(self.start_idx, n)

        # Shuffle indices in buffer for approximate shuffling
        if self.shuffle_buffer > 0:
            self.rng.shuffle(indices)

        for idx in indices:
            lo = self.offsets[idx]
            hi = self.offsets[idx + 1] if idx + 1 < len(self.offsets) else len(self.all_ids)
            seq = self.all_ids[lo:hi]
            yield torch.from_numpy(seq.astype(np.int64))


def collate_tokenized(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad to max length in batch, aligned to 8."""
    ALIGN = 8
    src_max = max(t.size(0) for t in batch)
    src_max = ((src_max + ALIGN - 1) // ALIGN) * ALIGN

    padded = torch.full((len(batch), src_max), PAD_ID, dtype=torch.long)
    for i, t in enumerate(batch):
        padded[i, :t.size(0)] = t

    mask = (padded != PAD_ID)
    return padded, padded.clone(), mask, mask  # [src, tgt placeholder, src_mask, tgt_mask]


def collate_paired(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    """Collate paired src+tgt batches."""
    batch_src, batch_tgt = zip(*batch)
    ALIGN = 8
    src_max = max(t.size(0) for t in batch_src)
    tgt_max = max(t.size(0) for t in batch_tgt)
    src_max = ((src_max + ALIGN - 1) // ALIGN) * ALIGN
    tgt_max = ((tgt_max + ALIGN - 1) // ALIGN) * ALIGN

    src_padded = torch.full((len(batch_src), src_max), PAD_ID, dtype=torch.long)
    tgt_padded = torch.full((len(batch_tgt), tgt_max), PAD_ID, dtype=torch.long)
    for i in range(len(batch_src)):
        src_padded[i, :batch_src[i].size(0)] = batch_src[i]
        tgt_padded[i, :batch_tgt[i].size(0)] = batch_tgt[i]

    src_mask = (src_padded != PAD_ID)
    tgt_mask = (tgt_padded != PAD_ID)
    return src_padded, tgt_padded, src_mask, tgt_mask


class PairedTokenizedDataset(IterableDataset):
    """Yield (src_tensor, tgt_tensor) pairs from pre-tokenized files."""

    def __init__(self, src_ids: str, src_offsets: str, src_lengths: str,
                 tgt_ids: str, tgt_offsets: str, tgt_lengths: str,
                 shuffle_buffer: int = 20000, seed: int = 42,
                 start_idx: int = 0, end_idx: int | None = None,
                 max_len: int = 128):
        super().__init__()
        self.max_len = max_len
        self.src_ids = np.load(src_ids, mmap_mode="r")
        self.src_offsets = np.load(src_offsets, mmap_mode="r")
        self.src_lengths = np.load(src_lengths, mmap_mode="r")
        self.tgt_ids = np.load(tgt_ids, mmap_mode="r")
        self.tgt_offsets = np.load(tgt_offsets, mmap_mode="r")
        self.tgt_lengths = np.load(tgt_lengths, mmap_mode="r")
        self.shuffle_buffer = shuffle_buffer
        self.rng = np.random.default_rng(seed)
        self.start_idx = start_idx
        self.end_idx = end_idx if end_idx is not None else len(self.src_lengths)
        self.end_idx = min(self.end_idx, len(self.src_lengths), len(self.tgt_lengths))

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        n = self.end_idx
        indices = np.arange(self.start_idx, n)

        # Partition across DataLoader workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            per_worker = len(indices) // worker_info.num_workers
            lo = worker_info.id * per_worker
            hi = lo + per_worker if worker_info.id < worker_info.num_workers - 1 else len(indices)
            indices = indices[lo:hi]

        # Bucket by source length: sort buffer in chunks so the DataLoader
        # collate groups similar-length sequences together → less padding.
        bucket_size = max(self.shuffle_buffer, 128)  # at least batch_size to avoid zero-step

        for start in range(0, len(indices), bucket_size):
            chunk = indices[start:start + bucket_size]
            # Sort by source length via argsort on the lengths array
            chunk_lens = self.src_lengths[chunk]
            sorted_order = np.argsort(chunk_lens)
            chunk = chunk[sorted_order]
            # Light shuffle within length bands for randomness
            band = max(1, len(chunk) // 50)
            for b in range(0, len(chunk), band):
                self.rng.shuffle(chunk[b:b + band])

            for idx in chunk:
                slo = self.src_offsets[idx]
                shi = self.src_offsets[idx + 1] if idx + 1 < len(self.src_offsets) else len(self.src_ids)
                tlo = self.tgt_offsets[idx]
                thi = self.tgt_offsets[idx + 1] if idx + 1 < len(self.tgt_offsets) else len(self.tgt_ids)

                src_seq = self.src_ids[slo:shi][:self.max_len]
                tgt_seq = self.tgt_ids[tlo:thi][:self.max_len]
                yield (torch.from_numpy(src_seq.astype(np.int64)),
                       torch.from_numpy(tgt_seq.astype(np.int64)))


def build_tokenized_dataloaders(config) -> tuple[DataLoader, DataLoader]:
    data_dir = config.data_dir / ".." / "data_tokenized" if str(config.data_dir) == "data" else config.data_dir / "tokenized"
    # resolve relative path
    from pathlib import Path
    data_dir = Path("data_tokenized")

    dl_kwargs = dict(
        batch_size=config.batch_size,
        collate_fn=collate_paired,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False,
    )
    if config.num_workers > 0:
        dl_kwargs["prefetch_factor"] = 2

    train_ds = PairedTokenizedDataset(
        str(data_dir / "train_src_ids.npy"),
        str(data_dir / "train_src_offsets.npy"),
        str(data_dir / "train_src_lengths.npy"),
        str(data_dir / "train_tgt_ids.npy"),
        str(data_dir / "train_tgt_offsets.npy"),
        str(data_dir / "train_tgt_lengths.npy"),
        shuffle_buffer=20000, seed=config.seed,
        end_idx=config.max_train_samples if config.max_train_samples > 0 else None,
        max_len=config.max_seq_len,
    )

    val_ds = PairedTokenizedDataset(
        str(data_dir / "val_src_ids.npy"),
        str(data_dir / "val_src_offsets.npy"),
        str(data_dir / "val_src_lengths.npy"),
        str(data_dir / "val_tgt_ids.npy"),
        str(data_dir / "val_tgt_offsets.npy"),
        str(data_dir / "val_tgt_lengths.npy"),
        shuffle_buffer=2000, seed=config.seed,
        max_len=config.max_seq_len,
    )

    train_loader = DataLoader(train_ds, **dl_kwargs)
    val_loader = DataLoader(val_ds, **dl_kwargs)
    total_train_pairs = train_ds.end_idx - train_ds.start_idx
    return train_loader, val_loader, total_train_pairs
