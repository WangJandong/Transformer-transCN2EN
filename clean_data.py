"""Clean parallel corpus: dedup, filter, normalize.

Produces data/clean/train.{zh,en} from the raw files.

Filters (in order):
  1. Drop empty / single-char lines
  2. Drop pairs with extreme character-length ratio
  3. Drop language-mismatched pairs (Latin-heavy zh side, CJK-heavy en side)
  4. Deduplicate exact pairs
  5. Drop pairs where either side exceeds a max-token budget
  6. Drop sentences with too-high special-char ratio

Usage:
    python clean_data.py                 # run with defaults
    python clean_data.py --full          # clean the whole 19M dataset (slow!)
    python clean_data.py --sample 2M     # clean first 2M lines only
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

import sentencepiece as spm

# ── rule predicates ───────────────────────────────────────────────

_CJK_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿]")
_LATIN_RE = re.compile(r"[a-zA-Z]")
_PRINTABLE_ASCII = set(range(0x20, 0x7F))
_PRINTABLE_ASCII.add(0x0A)  # newline


def _frac_cjk(text: str) -> float:
    if not text:
        return 0.0
    return len(_CJK_RE.findall(text)) / len(text)


def _frac_latin(text: str) -> float:
    if not text:
        return 0.0
    return len(_LATIN_RE.findall(text)) / len(text)


def _frac_non_printable(text: str) -> float:
    if not text:
        return 0.0
    non_print = sum(1 for ch in text if ord(ch) not in _PRINTABLE_ASCII
                    and not (0x4e00 <= ord(ch) <= 0x9fff)
                    and not (0x3400 <= ord(ch) <= 0x4dbf))
    return non_print / len(text)


# ── filter stats ──────────────────────────────────────────────────
_EMPTY = 0
_RATIO = 0
_LANG_MISMATCH = 0
_DUP = 0
_TOO_LONG = 0
_WEIRD_CHARS = 0
_KEPT = 0


def _drop(reason: str) -> None:
    global _EMPTY, _RATIO, _LANG_MISMATCH, _DUP, _TOO_LONG, _WEIRD_CHARS
    if reason == "empty":
        _EMPTY += 1
    elif reason == "ratio":
        _RATIO += 1
    elif reason == "lang":
        _LANG_MISMATCH += 1
    elif reason == "dup":
        _DUP += 1
    elif reason == "long":
        _TOO_LONG += 1
    elif reason == "weird":
        _WEIRD_CHARS += 1


# ── main ──────────────────────────────────────────────────────────

def clean(
    src_in: Path,
    tgt_in: Path,
    src_out: Path,
    tgt_out: Path,
    sp,
    max_lines: int = 0,
    max_tokens: int = 200,
    max_len_ratio: float = 3.0,
    min_len_ratio: float = 0.33,
    min_chars: int = 4,
    max_special_frac: float = 0.3,
    dedup: bool = True,
):
    global _KEPT
    src_out.parent.mkdir(parents=True, exist_ok=True)

    seen: set[tuple[str, str]] = set()
    _KEPT = 0
    total = 0

    with open(src_in, encoding="utf-8") as fz, \
         open(tgt_in, encoding="utf-8") as fe, \
         open(src_out, "w", encoding="utf-8") as fz_out, \
         open(tgt_out, "w", encoding="utf-8") as fe_out:

        for zh, en in zip(fz, fe):
            total += 1
            if max_lines and total > max_lines:
                break
            if total % 2_000_000 == 0:
                print(f"  processed {total/1e6:.0f}M … kept {_KEPT:,}", flush=True)

            zh = zh.strip()
            en = en.strip()

            # ── 1. empty / too short ──
            if len(zh) < min_chars or len(en) < min_chars:
                _drop("empty")
                continue

            # ── 2. length ratio (char-level, fast) ──
            zc, ec = len(zh), len(en)
            ratio = zc / ec if ec > 0 else 999
            if ratio < min_len_ratio or ratio > max_len_ratio:
                _drop("ratio")
                continue

            # ── 3. language detection ──
            fc_zh = _frac_cjk(zh)
            fl_zh = _frac_latin(zh)
            fc_en = _frac_cjk(en)
            if fl_zh > 0.5 and fc_zh < 0.3:
                _drop("lang")  # source looks like English
                continue
            if fc_en > 0.3:
                _drop("lang")  # target contains substantial Chinese
                continue

            # ── 4. weird characters ──
            if _frac_non_printable(zh) > max_special_frac or \
               _frac_non_printable(en) > max_special_frac:
                _drop("weird")
                continue

            # ── 5. token budget ──
            z_tok = len(sp.encode(zh, out_type=int))
            e_tok = len(sp.encode(en, out_type=int))
            if z_tok > max_tokens or e_tok > max_tokens:
                _drop("long")
                continue

            # ── 6. dedup ──
            if dedup:
                pair = (zh, en)
                if pair in seen:
                    _drop("dup")
                    continue
                seen.add(pair)

            fz_out.write(zh + "\n")
            fe_out.write(en + "\n")
            _KEPT += 1

    # ── report ──
    kept_pct = _KEPT / max(total, 1) * 100
    print(f"\n{'='*55}")
    print(f"  CLEANING REPORT")
    print(f"{'='*55}")
    print(f"  Input pairs:         {total:>10,}")
    print(f"  Kept:                {_KEPT:>10,}  ({kept_pct:.1f}%)")
    print(f"  {'─'*45}")
    print(f"  Dropped:")
    print(f"    empty/short:       {_EMPTY:>10,}")
    print(f"    bad length ratio:  {_RATIO:>10,}")
    print(f"    language mismatch: {_LANG_MISMATCH:>10,}")
    print(f"    duplicates:        {_DUP:>10,}")
    print(f"    too long (>tok):   {_TOO_LONG:>10,}")
    print(f"    weird characters:  {_WEIRD_CHARS:>10,}")
    total_dropped = total - _KEPT
    print(f"  {'─'*45}")
    print(f"  Total dropped:       {total_dropped:>10,}  ({total_dropped/max(total,1)*100:.1f}%)")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="process all 19M lines")
    parser.add_argument("--sample", type=int, default=0, help="process first N lines")
    parser.add_argument("--no_dedup", action="store_true")
    parser.add_argument("--max_tokens", type=int, default=200)
    args = parser.parse_args()

    from config import Config
    config = Config()

    sp = spm.SentencePieceProcessor()
    sp.load(f"{config.sp_model_prefix}.model")

    data_dir = config.data_dir
    out_dir = data_dir / "clean"
    max_lines = args.sample if args.sample > 0 else (0 if args.full else 2_000_000)

    if max_lines:
        print(f"Cleaning first {max_lines/1e6:.1f}M lines (use --full for all 19M)")

    clean(
        src_in=data_dir / f"train.{config.src_lang}",
        tgt_in=data_dir / f"train.{config.tgt_lang}",
        src_out=out_dir / f"train.{config.src_lang}",
        tgt_out=out_dir / f"train.{config.tgt_lang}",
        sp=sp,
        max_lines=max_lines,
        max_tokens=args.max_tokens,
        dedup=not args.no_dedup,
    )
