"""Analyse training data quality and suggest filtering rules.

Run separately — reads the full training set once, writes stats + filter list.
"""
from __future__ import annotations
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import sentencepiece as spm

# ── helper predicates ─────────────────────────────────────────────
# CJK Unicode ranges
_CJK_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿]")
# Latin letters (English side should dominate with these)
_LATIN_RE = re.compile(r"[a-zA-Z]")


def frac_cjk(text: str) -> float:
    """Fraction of characters that are CJK."""
    if not text:
        return 0.0
    return len(_CJK_RE.findall(text)) / len(text)


def frac_latin(text: str) -> float:
    """Fraction of characters that are Latin (a-zA-Z)."""
    if not text:
        return 0.0
    return len(_LATIN_RE.findall(text)) / len(text)


def is_empty_or_single(text: str) -> bool:
    return len(text.strip()) < 2


FULLWIDTH_PUNCT = set("，。！？：；“”‘’（）【】《》—…、")


def detect_anomalous_chars(text: str) -> list[str]:
    """Return rare fullwidth / non-standard chars that look like noise."""
    unusual = []
    for ch in text:
        if ch in FULLWIDTH_PUNCT:
            continue
        cp = ord(ch)
        if 0x4e00 <= cp <= 0x9fff:   # CJK unified
            continue
        if 0x20 <= cp <= 0x7e:       # ASCII printable
            continue
        if cp in (0x0a, 0x0d):       # newline / CR
            continue
        # common fullwidth
        if 0xff00 <= cp <= 0xffef:
            continue
        if 0x3000 <= cp <= 0x303f:   # CJK punctuation
            continue
        unusual.append(f"U+{cp:04X}")
    return unusual[:10]  # cap


# ── analysis ──────────────────────────────────────────────────────
def analyse(config, max_lines: int = 0):
    src_path = config.data_dir / f"train.{config.src_lang}"
    tgt_path = config.data_dir / f"train.{config.tgt_lang}"

    sp = spm.SentencePieceProcessor()
    sp.load(f"{config.sp_model_prefix}.model")

    src_lens: list[int] = []
    tgt_lens: list[int] = []
    src_tok_lens: list[int] = []
    tgt_tok_lens: list[int] = []
    len_ratios: list[float] = []
    cjk_on_en: list[tuple[int, str, str]] = []
    latin_on_zh: list[tuple[int, str, str]] = []
    empty_lines: list[int] = []
    weird_chars: list[tuple[int, str, str]] = []
    dup_pairs: set[tuple[str, str]] = set()
    dup_count: int = 0
    n: int = 0

    with open(src_path, encoding="utf-8") as fz, open(tgt_path, encoding="utf-8") as fe:
        for i, (zh, en) in enumerate(zip(fz, fe)):
            zh = zh.strip()
            en = en.strip()

            if is_empty_or_single(zh) or is_empty_or_single(en):
                empty_lines.append(i)
                continue

            n += 1
            if max_lines and n >= max_lines:
                break

            zc = len(zh)
            ec = len(en)
            src_lens.append(zc)
            tgt_lens.append(ec)

            # token lengths
            zt = len(sp.encode(zh, out_type=int))
            et = len(sp.encode(en, out_type=int))
            src_tok_lens.append(zt)
            tgt_tok_lens.append(et)

            # length ratio
            if ec > 0 and zc > 0:
                len_ratios.append(zt / et if et > 0 else zt)

            # language mismatch
            fc_zh = frac_cjk(zh)
            fl_zh = frac_latin(zh)
            fc_en = frac_cjk(en)
            fl_en = frac_latin(en)

            if fc_en > 0.3 and fl_en < 0.3:
                cjk_on_en.append((i, zh[:60], en[:60]))
            if fl_zh > 0.5 and fc_zh < 0.3:
                latin_on_zh.append((i, zh[:60], en[:60]))

            # duplicates
            pair = (zh, en)
            if pair in dup_pairs:
                dup_count += 1
            else:
                dup_pairs.add(pair)

            # weird characters
            anom = detect_anomalous_chars(zh) or detect_anomalous_chars(en)
            if anom and len(weird_chars) < 20:
                weird_chars.append((i, zh[:60], en[:60]))

            # progress
            if i % 2_000_000 == 0 and i > 0:
                print(f"  processed {i/1e6:.0f}M lines …", flush=True)

    print(f"\n{'='*60}")
    print(f"  DATA QUALITY REPORT  —  {n:,} pairs analysed")
    print(f"{'='*60}")

    # ── length stats ──
    print(f"\n  ── Character lengths ──")
    src_arr, tgt_arr = np.array(src_lens), np.array(tgt_lens)
    for label, arr in [("zh source", src_arr), ("en target", tgt_arr)]:
        print(f"    {label:12s}: mean={arr.mean():.0f}  median={np.median(arr):.0f}  "
              f"p95={np.percentile(arr,95):.0f}  p99={np.percentile(arr,99):.0f}  max={arr.max()}")

    print(f"\n  ── BPE token lengths ──")
    ztok, etok = np.array(src_tok_lens), np.array(tgt_tok_lens)
    for label, arr in [("zh source", ztok), ("en target", etok)]:
        print(f"    {label:12s}: mean={arr.mean():.0f}  median={np.median(arr):.0f}  "
              f"p95={np.percentile(arr,95):.0f}  p99={np.percentile(arr,99):.0f}  max={arr.max()}")

    # ── length ratio ──
    ratios = np.array(len_ratios)
    print(f"\n  ── Length ratio (zh_tok / en_tok) ──")
    print(f"    mean={ratios.mean():.2f}  median={np.median(ratios):.2f}  "
          f"p5={np.percentile(ratios,5):.2f}  p95={np.percentile(ratios,95):.2f}")
    extreme = (ratios < 0.2) | (ratios > 5.0)
    print(f"    extreme (<0.2 or >5.0): {extreme.sum():,} ({extreme.sum()/len(ratios)*100:.2f}%)")

    # ── empty ──
    print(f"\n  ── Empty/single-char lines ──")
    print(f"    {len(empty_lines):,} pairs")

    # ── language mismatch ──
    print(f"\n  ── Language mismatch ──")
    print(f"    CJK-heavy en (>30% CJK): {len(cjk_on_en):,}")
    for i, z, e in cjk_on_en[:5]:
        print(f"      [{i}] zh={z}  |  en={e}")
    print(f"    Latin-heavy zh (>50% Latin): {len(latin_on_zh):,}")
    for i, z, e in latin_on_zh[:5]:
        print(f"      [{i}] zh={z}  |  en={e}")

    # ── duplicates ──
    print(f"\n  ── Duplicates ──")
    print(f"    exact duplicate pairs: {dup_count:,} ({dup_count/n*100:.2f}%)")

    # ── weird chars ──
    print(f"\n  ── Non-standard characters ──")
    for i, z, e in weird_chars[:10]:
        print(f"      [{i}] {z}  |  {e}")

    # ── length bucketed distribution ──
    print(f"\n  ── Token-length buckets (src) ──")
    buckets = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, 160), (160, 256)]
    for lo, hi in buckets:
        cnt = ((ztok >= lo) & (ztok < hi)).sum()
        print(f"    [{lo:3d}, {hi:3d}): {cnt:>8,}  ({cnt/len(ztok)*100:5.1f}%)")

    # ── recommendations ──
    print(f"\n  {'='*60}")
    print(f"  FILTERING RECOMMENDATIONS")
    print(f"  {'='*60}")
    recs = [
        ("empty / too-short lines", len(empty_lines)),
        ("extreme length ratio (<0.2 or >5.0)", extreme.sum()),
        ("CJK on English side", len(cjk_on_en)),
        ("Latin-heavy zh (bad alignment)", len(latin_on_zh)),
        ("exact duplicate pairs", dup_count),
    ]
    total_bad = sum(r[1] for r in recs)
    print(f"    Total flagged: ~{total_bad:,} ({total_bad/n*100:.1f}% of data)")
    for label, cnt in recs:
        print(f"    - {label}: {cnt:,} ({cnt/n*100:.2f}%)")

    return {
        "n": n,
        "empty": len(empty_lines),
        "extreme_ratio": int(extreme.sum()),
        "cjk_on_en": len(cjk_on_en),
        "latin_on_zh": len(latin_on_zh),
        "duplicates": dup_count,
        "src_mean_tok": float(ztok.mean()),
        "tgt_mean_tok": float(etok.mean()),
        "ratio_median": float(np.median(ratios)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=5_000_000, help="lines to scan (0=all)")
    args = parser.parse_args()

    from config import Config
    config = Config()
    analyse(config, max_lines=args.sample)
