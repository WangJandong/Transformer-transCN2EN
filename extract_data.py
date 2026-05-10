"""Rebuild clean train/val/test from the original CSV using proper csv.reader parsing.

Optimised single-pass:
  - No SentencePiece calls (char-level filters only — fast).
  - zlib.crc32 for hash-based split (deterministic, ~10× faster than md5).
  - 8MB write buffers + batched writes.

Usage:
    python extract_data.py                     # default filters, 2M train
    python extract_data.py --train_lines 0     # all rows (full 18.7M)
    python extract_data.py --no_filter         # raw extraction
"""
from __future__ import annotations
import argparse
import csv
import re
import zlib
from pathlib import Path

_CJK_RE = re.compile(r"[一-鿿]")
_LATIN_RE = re.compile(r"[a-zA-Z]")


def _frac_cjk(text: str) -> float:
    if not text:
        return 0.0
    return len(_CJK_RE.findall(text)) / max(len(text), 1)


def _frac_latin(text: str) -> float:
    if not text:
        return 0.0
    return len(_LATIN_RE.findall(text)) / max(len(text), 1)


# Output buffer — flush every N writes to avoid per-line syscall overhead
BUF_SIZE = 10_000


def extract(train_limit: int, no_filter: bool):
    csv_path = ("WMT-Chinese-to-English-Machine-Translation-Training-Corpus-new/"
                "wmt_zh_en_training_corpus.csv")
    out_dir = Path("data")

    MIN_CHARS = 4
    MAX_CHARS = 400
    MIN_LEN_RATIO = 0.25
    MAX_LEN_RATIO = 4.0
    MIN_ZH_CJK = 0.08
    MIN_EN_LATIN = 0.25

    train_ratio = 0.98
    val_ratio = 0.01
    test_ratio = 0.01

    out_dir.mkdir(parents=True, exist_ok=True)

    # Big write buffers (8MB)
    f_train_zh = open(out_dir / "train.zh", "w", encoding="utf-8", buffering=8 * 1024 * 1024)
    f_train_en = open(out_dir / "train.en", "w", encoding="utf-8", buffering=8 * 1024 * 1024)
    f_val_zh   = open(out_dir / "val.zh",   "w", encoding="utf-8", buffering=8 * 1024 * 1024)
    f_val_en   = open(out_dir / "val.en",   "w", encoding="utf-8", buffering=8 * 1024 * 1024)
    f_test_zh  = open(out_dir / "test.zh",  "w", encoding="utf-8", buffering=8 * 1024 * 1024)
    f_test_en  = open(out_dir / "test.en",  "w", encoding="utf-8", buffering=8 * 1024 * 1024)

    # Per-file line buffers
    buf_train_zh: list[str] = []; buf_train_en: list[str] = []
    buf_val_zh:   list[str] = []; buf_val_en:   list[str] = []
    buf_test_zh:  list[str] = []; buf_test_en:  list[str] = []

    def _flush(buf_src, buf_tgt, f_src, f_tgt):
        if buf_src:
            f_src.write("".join(buf_src))
            f_tgt.write("".join(buf_tgt))
            buf_src.clear()
            buf_tgt.clear()

    # stats
    total = 0
    kept = 0
    drops = {"empty": 0, "ratio": 0, "lang": 0, "dup": 0}
    train_n = val_n = test_n = 0
    train_written = 0
    seen: set[tuple[str, str]] = set()

    # For CRC32 hash: encode to bytes once for both dedup key and hash
    sep = "\t".encode()

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip "0,1"

        for zh_raw, en_raw in reader:
            total += 1
            zh = zh_raw.strip()
            en = en_raw.strip()

            if no_filter:
                if not zh or not en:
                    continue
            else:
                # 1. char length
                zc, ec = len(zh), len(en)
                if zc < MIN_CHARS or ec < MIN_CHARS:
                    drops["empty"] += 1; continue
                if zc > MAX_CHARS or ec > MAX_CHARS:
                    drops["empty"] += 1; continue

                # 2. length ratio
                ratio = zc / max(ec, 1)
                if ratio < MIN_LEN_RATIO or ratio > MAX_LEN_RATIO:
                    drops["ratio"] += 1; continue

                # 3. language check (fast regex)
                if _frac_cjk(zh) < MIN_ZH_CJK:
                    drops["lang"] += 1; continue
                if _frac_latin(en) < MIN_EN_LATIN:
                    drops["lang"] += 1; continue

                # 4. dedup
                pair = (zh, en)
                if pair in seen:
                    drops["dup"] += 1; continue
                seen.add(pair)

            kept += 1

            # ── hash-based split (crc32, fast + deterministic) ──
            h = zlib.crc32(zh.encode()) ^ zlib.crc32(en.encode())
            h = h % 100

            if h < train_ratio * 100:
                train_n += 1
                if train_limit == 0 or train_written < train_limit:
                    buf_train_zh.append(zh + "\n")
                    buf_train_en.append(en + "\n")
                    train_written += 1
                    if len(buf_train_zh) >= BUF_SIZE:
                        _flush(buf_train_zh, buf_train_en, f_train_zh, f_train_en)
            elif h < (train_ratio + val_ratio) * 100:
                val_n += 1
                buf_val_zh.append(zh + "\n")
                buf_val_en.append(en + "\n")
                if len(buf_val_zh) >= BUF_SIZE:
                    _flush(buf_val_zh, buf_val_en, f_val_zh, f_val_en)
            else:
                test_n += 1
                buf_test_zh.append(zh + "\n")
                buf_test_en.append(en + "\n")
                if len(buf_test_zh) >= BUF_SIZE:
                    _flush(buf_test_zh, buf_test_en, f_test_zh, f_test_en)

            if total % 5_000_000 == 0:
                print(f"  processed {total/1e6:.0f}M … "
                      f"kept={kept:,} train={train_written:,} val={val_n:,} test={test_n:,}",
                      flush=True)

    # Final flush
    _flush(buf_train_zh, buf_train_en, f_train_zh, f_train_en)
    _flush(buf_val_zh,   buf_val_en,   f_val_zh,   f_val_en)
    _flush(buf_test_zh,  buf_test_en,  f_test_zh,  f_test_en)

    f_train_zh.close(); f_train_en.close()
    f_val_zh.close();   f_val_en.close()
    f_test_zh.close();  f_test_en.close()

    kept_pct = kept / max(total, 1) * 100
    drop_total = sum(drops.values())

    print(f"\n{'='*55}")
    print(f"  EXTRACTION REPORT")
    print(f"{'='*55}")
    print(f"  CSV rows read:     {total:>12,}")
    print(f"  Kept:              {kept:>12,}  ({kept_pct:.1f}%)")
    if not no_filter:
        print(f"\n  Dropped:")
        for k, v in drops.items():
            print(f"    {k:<14s} {v:>10,}  ({v/max(total,1)*100:5.2f}%)")
        print(f"    {'─'*30}")
        print(f"    {'total':<14s} {drop_total:>10,}  ({drop_total/max(total,1)*100:5.2f}%)")
    print(f"\n  Split (assigned):")
    print(f"    train: {train_n:>12,}  ({train_n/max(kept,1)*100:.1f}%)")
    print(f"    val:   {val_n:>12,}  ({val_n/max(kept,1)*100:.1f}%)")
    print(f"    test:  {test_n:>12,}  ({test_n/max(kept,1)*100:.1f}%)")
    print(f"\n  Train written:     {train_written:>12,}")
    print(f"{'='*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild data/ from clean CSV")
    parser.add_argument("--train_lines", type=int, default=2_000_000,
                        help="max train lines (0 = unlimited)")
    parser.add_argument("--no_filter", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    extract(train_limit=args.train_lines, no_filter=args.no_filter)
