"""
Clean training and benchmark data: remove noise artifacts.

Fixes:
  1. @USER / @user → removed (Twitter mentions don't affect label)
  2. RT @xxx → removed (retweet markers)
  3. <LF> → space (newline tokens from preprocessed datasets)
  4. https://... / http://... → [URL] for spam, removed for other labels
  5. <URL> → same as above
  6. &amp; &lt; &gt; → decoded
  7. Multiple spaces/tabs/newlines → single space
  8. Remove rows where text becomes empty or too short (< 10 chars)

Usage:
  python -m generate.clean_data
"""

import csv
import re
import sys
from pathlib import Path

DATA_DIR = Path("/home/ninini/Agents/Generate_data/data_v2")
BENCHMARK_PATH = Path("/home/ninini/Agents/Generate_data/data_csv/safety_benchmark.csv")

COUNTRIES = ["BR", "ID", "MX", "SA", "SG", "TH", "TR", "ZA"]


def clean_text(text: str, label: str = "") -> str:
    """Clean a single text. Returns cleaned text or empty string if too noisy."""

    # 1. URL handling — keep [URL] placeholder for spam, strip for others
    is_spam = label == "base_spam_fraud"

    # Remove Twitter shortlinks
    text = re.sub(r'https?://t\.co/\S+', '', text, flags=re.IGNORECASE)

    # Remove full URLs
    if is_spam:
        text = re.sub(r'https?://\S+', '[URL]', text, flags=re.IGNORECASE)
        text = re.sub(r'www\.\S+', '[URL]', text, flags=re.IGNORECASE)
        text = re.sub(r'<URL>', '[URL]', text, flags=re.IGNORECASE)
    else:
        text = re.sub(r'https?://\S+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'www\.\S+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<URL>', '', text, flags=re.IGNORECASE)

    # 2. @USER mentions — remove
    text = re.sub(r'@\s*USER\b', '', text, flags=re.IGNORECASE)

    # 3. RT markers — remove prefix
    text = re.sub(r'\bRT\s*@?\w*:?\s*', '', text)

    # 4. <LF> newline tokens — space
    text = text.replace('<LF>', ' ')

    # 5. HTML entities
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&#39;', "'")

    # 6. Hashtags — keep text but clean # if alone
    # (don't strip hashtags — they carry meaning in some labels)

    # 7. Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # 8. Trim
    text = text.strip()

    # 9. Remove leading/trailing special chars that don't add meaning
    text = re.sub(r'^[,\s.…]+', '', text)
    text = re.sub(r'[,\s.…]+$', '', text)

    return text


def clean_csv(input_path: Path, output_path: Path, is_training: bool = True):
    """Clean a CSV file."""
    with open(input_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    cleaned = []
    removed_empty = 0
    removed_short = 0

    for r in rows:
        text = r.get("text", "")
        label = r.get("label_name", r.get("label", "")) if is_training else r.get("label", "")
        original = text

        cleaned_text = clean_text(text, label)

        # Skip if empty or too short
        if not cleaned_text:
            removed_empty += 1
            continue
        if len(cleaned_text) < 10:
            removed_short += 1
            continue

        r["text"] = cleaned_text
        cleaned.append(r)

    # Write back
    fieldnames = cleaned[0].keys() if cleaned else []
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cleaned)

    return {
        "input": len(rows),
        "output": len(cleaned),
        "removed_empty": removed_empty,
        "removed_short": removed_short,
    }


def main():
    print("Cleaning training data...")
    grand_input = 0
    grand_output = 0
    grand_removed = 0

    for cc in COUNTRIES:
        input_path = DATA_DIR / f"train_{cc}.csv"
        output_path = DATA_DIR / f"train_{cc}.csv"
        if not input_path.exists():
            print(f"  {cc}: not found, skipping")
            continue

        stats = clean_csv(input_path, output_path, is_training=True)
        removed = stats["removed_empty"] + stats["removed_short"]
        print(f"  {cc}: {stats['input']:,} → {stats['output']:,} "
              f"(removed {removed}: {stats['removed_empty']} empty, {stats['removed_short']} too short)")
        grand_input += stats["input"]
        grand_output += stats["output"]
        grand_removed += removed

    print(f"\nTraining total: {grand_input:,} → {grand_output:,} (removed {grand_removed})")

    # Clean benchmark
    if BENCHMARK_PATH.exists():
        print(f"\nCleaning benchmark...")
        stats = clean_csv(BENCHMARK_PATH, BENCHMARK_PATH, is_training=False)
        removed = stats["removed_empty"] + stats["removed_short"]
        print(f"  Benchmark: {stats['input']:,} → {stats['output']:,} "
              f"(removed {removed}: {stats['removed_empty']} empty, {stats['removed_short']} too short)")

    # Show some before/after examples
    print(f"\n=== Before/After Examples ===")
    samples = [
        "@USER @USER meu amor, com bolo de rolo...",
        "@USER كفو يا #الملك_سلمان <LF>قفلت على القطريين",
        "RT @USER: رفعت كفي نحو عطفك داعياً <LF>وعلمت أنك لا ترد دعائي",
        "Check this out https://t.co/abc123 for free money!",
        "Get your prize now www.scam.com/vip",
        "@USER الي يبيع الكليجا لابس كمام 😷<LF>من ريحة اديوسهن",
    ]
    for s in samples:
        cleaned = clean_text(s, "")
        print(f"  BEFORE: {s[:100]}")
        print(f"  AFTER:  {cleaned[:100]}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
