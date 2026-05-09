"""
Integrate external annotated data (new_annotated_data.csv) into the existing
Frozen Base + Country LoRA taxonomy.

External 7-category labels → internal base + local labels.
"""

import csv
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path("/home/ninini/Datasets_Nine_country")
OUT_DIR = Path("/home/ninini/Agents/Generate_data/output")

# Country folder → country code
FOLDER_TO_CC = {
    "Brazil": "BR",
    "Indonesia": "ID",
    "Mexico": "MX",
    "Saudi-Arabia": "SA",
    "Singapore": "SG",
    "South-Africa": "ZA",
    "Thailand": "TH",
    "Turkiye": "TR",
}

# Severity and detection_type for internal labels
LABEL_META = {
    "base_hate_speech_harassment": {"severity": "HIGH", "detection_type": "hybrid"},
    "base_violence_dangerous_behavior": {"severity": "HIGH", "detection_type": "contextual"},
    "base_sexual_content_child_safety": {"severity": "CRITICAL", "detection_type": "hybrid"},
    "base_spam_deceptive_practices": {"severity": "LOW", "detection_type": "keyword_sensitive"},
    "base_self_harm_suicide": {"severity": "HIGH", "detection_type": "contextual"},
    "local_br_structural_racism": {"severity": "CRITICAL", "detection_type": "hybrid"},
    "local_br_political_extremism": {"severity": "HIGH", "detection_type": "contextual"},
    "local_id_sara_violation": {"severity": "CRITICAL", "detection_type": "hybrid"},
    "local_id_pornography_slang": {"severity": "HIGH", "detection_type": "keyword_sensitive"},
    "local_mx_narco_culture": {"severity": "HIGH", "detection_type": "contextual"},
    "local_mx_gender_violence": {"severity": "HIGH", "detection_type": "contextual"},
    "local_sa_blasphemy_anti_islam": {"severity": "CRITICAL", "detection_type": "contextual"},
    "local_sa_immorality_lgbtq": {"severity": "HIGH", "detection_type": "keyword_sensitive"},
    "local_sa_anti_state": {"severity": "HIGH", "detection_type": "contextual"},
    "local_sg_racial_religious_harmony": {"severity": "CRITICAL", "detection_type": "contextual"},
    "local_sg_vulgarity_singlish": {"severity": "MEDIUM", "detection_type": "hybrid"},
    "local_th_lese_majeste": {"severity": "CRITICAL", "detection_type": "contextual"},
    "local_th_political_instigation": {"severity": "HIGH", "detection_type": "contextual"},
    "local_tr_insulting_state": {"severity": "HIGH", "detection_type": "contextual"},
    "local_tr_separatism_terror": {"severity": "HIGH", "detection_type": "hybrid"},
    "local_za_severe_racism": {"severity": "CRITICAL", "detection_type": "keyword_sensitive"},
    "local_za_xenophobia": {"severity": "HIGH", "detection_type": "contextual"},
}

# External label → (base_label, country_specific_map)
# country_specific_map: {CC: [local_label, ...]}
MAPPING = {
    "hate_speech": (
        "base_hate_speech_harassment",
        {
            "BR": ["local_br_structural_racism"],
            "ID": ["local_id_sara_violation"],
            "SG": ["local_sg_racial_religious_harmony"],
            "ZA": ["local_za_severe_racism", "local_za_xenophobia"],
            "SA": ["local_sa_blasphemy_anti_islam"],
            "MX": ["local_mx_gender_violence"],  # gender-based hate subset
        },
    ),
    "false_info": (
        "base_spam_deceptive_practices",
        {
            "BR": ["local_br_political_extremism"],
            "TH": ["local_th_political_instigation"],
        },
    ),
    "violence": (
        "base_violence_dangerous_behavior",
        {
            "MX": ["local_mx_narco_culture"],
            "TR": ["local_tr_separatism_terror"],
            "BR": ["local_br_political_extremism"],
        },
    ),
    "harassment": (
        "base_hate_speech_harassment",
        {
            "MX": ["local_mx_gender_violence"],
            "SG": ["local_sg_vulgarity_singlish"],
        },
    ),
    "obscenity": (
        "base_sexual_content_child_safety",
        {
            "ID": ["local_id_pornography_slang"],
            "SA": ["local_sa_immorality_lgbtq"],
        },
    ),
    "illegal": (
        "base_spam_deceptive_practices",
        {},  # no local labels currently
    ),
    "national_security": (
        None,  # no base label for this — purely country-specific
        {
            "TH": ["local_th_lese_majeste", "local_th_political_instigation"],
            "SA": ["local_sa_anti_state", "local_sa_blasphemy_anti_islam"],
            "TR": ["local_tr_insulting_state", "local_tr_separatism_terror"],
            "BR": ["local_br_political_extremism"],
        },
    ),
}

# Base labels for distributing negative samples
BASE_LABELS = [
    "base_violence_dangerous_behavior",
    "base_hate_speech_harassment",
    "base_sexual_content_child_safety",
    "base_self_harm_suicide",
    "base_spam_deceptive_practices",
]

EXTERNAL_LABELS = ["hate_speech", "false_info", "violence", "harassment", "obscenity", "illegal", "national_security"]


def _parse_float(val):
    """Parse a float value, handling empty strings."""
    if not val or not str(val).strip():
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def map_to_labels(cc: str, ext_labels: dict[str, float]) -> set[str]:
    """Map external label values to internal labels.

    Args:
        cc: Country code (BR, ID, MX, ...)
        ext_labels: {label_name: float_value} from the CSV row

    Returns:
        Set of internal label names
    """
    result = set()

    for ext_label, value in ext_labels.items():
        if value < 0.5:
            continue
        base_label, country_map = MAPPING.get(ext_label, (None, {}))

        # Add base label
        if base_label:
            result.add(base_label)

        # Add country-specific local labels
        for local_label in country_map.get(cc, []):
            result.add(local_label)

    return result


def _build_record(text: str, label: str, is_violation: bool, cc: str,
                  external_label: str, language: str = "") -> dict:
    """Build a JSONL record matching the existing format."""
    meta = LABEL_META.get(label, {"severity": "MEDIUM", "detection_type": "contextual"})
    return {
        "text": text.strip(),
        "label": label,
        "is_violation": is_violation,
        "severity": meta["severity"],
        "detection_type": meta["detection_type"],
        "language": language,
        "country_code": cc,
        "generation_strategy": "external_integration",
        "adversarial_technique": None,
        "metadata": {
            "source": "new_annotated_data",
            "external_label": external_label,
        },
    }


def load_existing_texts(cc: str) -> set[str]:
    """Load existing texts for a country to avoid duplicates."""
    existing = set()
    cc_dir = OUT_DIR / cc
    if not cc_dir.is_dir():
        return existing

    for fpath in sorted(cc_dir.glob("*.jsonl")):
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line.strip())
                    # Normalize for dedup
                    existing.add(d.get("text", "").strip())
                except json.JSONDecodeError:
                    continue
    return existing


def integrate_country(folder: str, cc: str, max_safe: int = 10000):
    """Integrate one country's new_annotated_data.csv.

    Args:
        folder: Country folder name in Datasets_Nine_country
        cc: Country code
        max_safe: Maximum number of safe (all-0) texts to import as negatives
    """
    csv_path = DATA_DIR / folder / "llm" / "deepseek-v4-flash" / "new_annotated_data.csv"
    if not csv_path.exists():
        logger.warning(f"No file at {csv_path}, skipping {cc}")
        return {}

    out_dir = OUT_DIR / cc
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load existing texts for dedup
    existing_texts = load_existing_texts(cc)
    logger.info(f"  Existing texts for {cc}: {len(existing_texts)}")

    stats = {
        "total_rows": 0,
        "violation_texts": 0,
        "safe_texts_imported": 0,
        "duplicates_skipped": 0,
        "records_written": 0,
        "per_label": defaultdict(lambda: {"pos": 0, "neg": 0}),
    }

    # Collect records per label before writing
    pending_records = defaultdict(list)

    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        safe_texts = []

        for row in reader:
            stats["total_rows"] += 1
            text = row.get("text", "").strip()
            if not text:
                continue

            # Skip if already in output
            if text in existing_texts:
                stats["duplicates_skipped"] += 1
                continue

            # Parse external labels
            ext_labels = {k: _parse_float(row.get(k, 0)) for k in EXTERNAL_LABELS}

            # Check if this is a violation (any label >= 0.5)
            is_violation = any(v >= 0.5 for v in ext_labels.values())

            if is_violation:
                stats["violation_texts"] += 1
                internal_labels = map_to_labels(cc, ext_labels)
                for lbl in internal_labels:
                    # Track which external label(s) triggered this
                    triggered_by = [k for k, v in ext_labels.items() if v >= 0.5]
                    rec = _build_record(text, lbl, True, cc, ",".join(triggered_by))
                    pending_records[lbl].append(rec)
                    stats["per_label"][lbl]["pos"] += 1
                    stats["records_written"] += 1
            else:
                safe_texts.append(text)

    # Randomly sample safe texts and assign to base labels as negatives
    if safe_texts and max_safe > 0:
        sample_size = min(max_safe, len(safe_texts))
        sampled = random.sample(safe_texts, sample_size)
        # Rotate through base labels
        for i, text in enumerate(sampled):
            lbl = BASE_LABELS[i % len(BASE_LABELS)]
            rec = _build_record(text, lbl, False, cc, "safe_general")
            pending_records[lbl].append(rec)
            stats["per_label"][lbl]["neg"] += 1
            stats["records_written"] += 1
            stats["safe_texts_imported"] += 1

    # Write all pending records
    for lbl, records in pending_records.items():
        fpath = out_dir / f"{lbl}.jsonl"
        with open(fpath, "a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return stats


def integrate_all(max_safe_per_country: int = 10000):
    """Integrate all 8 countries."""
    grand = {
        "total_rows": 0,
        "violation_texts": 0,
        "safe_texts_imported": 0,
        "duplicates_skipped": 0,
        "records_written": 0,
    }

    for folder, cc in sorted(FOLDER_TO_CC.items()):
        logger.info(f"Integrating {folder} ({cc})...")
        stats = integrate_country(folder, cc, max_safe=max_safe_per_country)

        for k in grand:
            grand[k] += stats.get(k, 0)

        # Print per-country summary
        pos = stats["violation_texts"]
        neg = stats["safe_texts_imported"]
        dup = stats["duplicates_skipped"]
        rec = stats["records_written"]
        logger.info(f"  {cc}: {stats['total_rows']:,} rows → {rec} records "
                    f"({pos} pos texts, {neg} neg texts, {dup} dup skipped)")

        for lbl, cnts in sorted(stats["per_label"].items()):
            if cnts["pos"] > 0 or cnts["neg"] > 0:
                logger.info(f"    {lbl}: +{cnts['pos']}pos +{cnts['neg']}neg")

    logger.info(f"\n=== GRAND TOTAL ===")
    logger.info(f"  Rows processed: {grand['total_rows']:,}")
    logger.info(f"  Violation texts: {grand['violation_texts']:,}")
    logger.info(f"  Safe texts imported: {grand['safe_texts_imported']:,}")
    logger.info(f"  Duplicates skipped: {grand['duplicates_skipped']:,}")
    logger.info(f"  Total records written: {grand['records_written']:,}")

    return grand


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    integrate_all(max_safe_per_country=10000)
