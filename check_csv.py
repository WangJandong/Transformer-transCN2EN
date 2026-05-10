"""Check the raw CSV format properly."""
import csv

total = 0
bad_field_count = 0
zh_quoted = 0
english_has_cjk = 0

with open("WMT-Chinese-to-English-Machine-Translation-Training-Corpus-new/wmt_zh_en_training_corpus.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        total += 1
        nf = len(row)
        if i < 10:
            print(f"  [{i}] fields={nf}: {[c[:50] for c in row]}")
        if nf != 2:
            bad_field_count += 1
        if nf >= 2:
            zh = row[0] if nf == 2 else row[1]
            en = row[1] if nf == 2 else row[2]
            if zh and zh[0] == '"':
                zh_quoted += 1
            if any('一' <= ch <= '鿿' for ch in en):
                english_has_cjk += 1
        if i % 5_000_000 == 0 and i > 0:
            print(f"  scanned {i/1e6:.0f}M … bad={bad_field_count} zh_quoted={zh_quoted}")

print(f"\nTotal CSV rows: {total:,}")
print(f"Rows with != 2 fields: {bad_field_count:,} ({bad_field_count/total*100:.1f}%)")
print(f"zh fields with leading quote: {zh_quoted:,}")
print(f"en fields with CJK chars: {english_has_cjk:,}")
