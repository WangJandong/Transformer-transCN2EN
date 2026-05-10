from __future__ import annotations
"""Pre-tokenize corpus → memory-mapped numpy arrays (streaming, low-RAM).

Token ids are written to a temp file every 500K lines using array('i').
Offsets + lengths accumulated in Python lists, saved as .npy at the end.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import os
import tempfile
from array import array
from pathlib import Path

import numpy as np
import sentencepiece as spm
from config import Config


def tokenize_stream(input_path: str, output_dir: Path, sp, max_len: int,
                    label: str, max_lines: int = 0):
    output_dir.mkdir(parents=True, exist_ok=True)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".bin", dir=str(output_dir))
    tmp_path = tmp.name

    lengths: list[int] = []
    arr = array("i")  # int32 accumulator
    n = 0

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            n += 1
            if max_lines and n > max_lines:
                break
            text = line.strip()
            if not text:
                continue
            ids = sp.encode(text, out_type=int)
            ids = [1] + ids[:max_len - 2] + [2]
            lengths.append(len(ids))
            arr.extend(ids)

            if len(arr) >= 10_000_000:  # ~40MB flush
                arr.tofile(tmp)
                arr = array("i")

            if n % 2_000_000 == 0:
                print(f"  {label}: {n/1e6:.0f}M lines …", flush=True)

    if len(arr) > 0:
        arr.tofile(tmp)
    tmp.close()

    # mmap → save as .npy
    total_ints = os.path.getsize(tmp_path) // 4
    ids_mmap = np.memmap(tmp_path, dtype=np.int32, mode="r", shape=(total_ints,))
    np.save(str(output_dir / f"{label}_ids.npy"), ids_mmap)
    del ids_mmap
    os.unlink(tmp_path)

    offsets = np.cumsum([0] + lengths, dtype=np.int64)
    np.save(str(output_dir / f"{label}_offsets.npy"), offsets)
    np.save(str(output_dir / f"{label}_lengths.npy"),
            np.array(lengths, dtype=np.int32))

    total_tokens = offsets[-1]
    print(f"  {label}: {len(lengths):,} lines → {total_tokens:,} tokens "
          f"(avg {total_tokens/max(len(lengths),1):.0f} tok/line)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["train", "val", "test", "all"])
    parser.add_argument("--max_lines", type=int, default=0)
    args = parser.parse_args()

    config = Config()
    sp = spm.SentencePieceProcessor()
    sp.load(f"{config.sp_model_prefix}.model")

    out_dir = Path("data_tokenized")
    print(f"Pre-tokenizing → {out_dir}/")

    for name in ["train", "val", "test"]:
        if args.dataset not in (name, "all"):
            continue
        src_path = config.data_dir / f"{name}.{config.src_lang}"
        tgt_path = config.data_dir / f"{name}.{config.tgt_lang}"
        if not src_path.exists():
            print(f"  SKIP {name}")
            continue

        print(f"\n  Processing {name} …")
        ml = args.max_lines if name == "train" and args.max_lines else 0
        tokenize_stream(str(src_path), out_dir, sp, config.max_seq_len, f"{name}_src", ml)
        tokenize_stream(str(tgt_path), out_dir, sp, config.max_seq_len, f"{name}_tgt", ml)

    # Verify
    for name in ["train", "val", "test"]:
        spath = out_dir / f"{name}_src_lengths.npy"
        tpath = out_dir / f"{name}_tgt_lengths.npy"
        if spath.exists() and tpath.exists():
            sl = len(np.load(spath, mmap_mode="r"))
            tl = len(np.load(tpath, mmap_mode="r"))
            assert sl == tl, f"{name}: src={sl} tgt={tl} mismatch!"
            print(f"  {name}: {sl:,} pairs OK")


if __name__ == "__main__":
    main()
