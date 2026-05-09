"""
Data balancing: downsample base_hate_speech_harassment positives and
add safe negative samples to local labels from external safe text pool.

Step 1: base_hate_speech_harassment pos 88K → ~6.4K (per country targets)
Step 2: each local label +2K general negatives from external safe pool
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

# Step 1: per-country positive target for base_hate_speech_harassment
HATE_SPEECH_POS_TARGET = {
    "BR": 800,
    "ID": 800,
    "MX": 800,
    "SA": 800,
    "SG": 1000,  # SG has less data overall
    "TH": 1000,  # TH has the least data
    "TR": 800,
    "ZA": 800,
}

# Step 2: negative samples to add per local label
LOCAL_NEG_TARGET = 2000

EXTERNAL_LABELS = ["hate_speech", "false_info", "violence", "harassment", "obscenity", "illegal", "national_security"]

# Local labels per country (from taxonomy)
COUNTRY_LOCAL_LABELS = {
    "BR": ["local_br_structural_racism", "local_br_political_extremism"],
    "ID": ["local_id_sara_violation", "local_id_pornography_slang"],
    "MX": ["local_mx_gender_violence", "local_mx_narco_culture"],
    "SA": ["local_sa_blasphemy_anti_islam", "local_sa_anti_state", "local_sa_immorality_lgbtq"],
    "SG": ["local_sg_racial_religious_harmony", "local_sg_vulgarity_singlish"],
    "TH": ["local_th_lese_majeste", "local_th_political_instigation"],
    "TR": ["local_tr_insulting_state", "local_tr_separatism_terror"],
    "ZA": ["local_za_severe_racism", "local_za_xenophobia"],
}

SEVERITY_MAP = {
    "local_br_structural_racism": "CRITICAL",
    "local_br_political_extremism": "HIGH",
    "local_id_sara_violation": "CRITICAL",
    "local_id_pornography_slang": "HIGH",
    "local_mx_gender_violence": "HIGH",
    "local_mx_narco_culture": "HIGH",
    "local_sa_blasphemy_anti_islam": "CRITICAL",
    "local_sa_anti_state": "HIGH",
    "local_sa_immorality_lgbtq": "HIGH",
    "local_sg_racial_religious_harmony": "CRITICAL",
    "local_sg_vulgarity_singlish": "MEDIUM",
    "local_th_lese_majeste": "CRITICAL",
    "local_th_political_instigation": "HIGH",
    "local_tr_insulting_state": "HIGH",
    "local_tr_separatism_terror": "HIGH",
    "local_za_severe_racism": "CRITICAL",
    "local_za_xenophobia": "HIGH",
}


def _parse_float(val):
    if not val or not str(val).strip():
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def step1_downsample_hate_speech():
    """Downsample base_hate_speech_harassment positive samples."""
    logger.info("=" * 60)
    logger.info("Step 1: Downsampling base_hate_speech_harassment")
    logger.info("=" * 60)

    total_before = 0
    total_after = 0
    total_pos_before = 0
    total_pos_after = 0

    for cc_dir in sorted(OUT_DIR.iterdir()):
        if not cc_dir.is_dir():
            continue
        cc = cc_dir.name
        if cc not in HATE_SPEECH_POS_TARGET:
            continue

        fpath = cc_dir / "base_hate_speech_harassment.jsonl"
        if not fpath.exists():
            logger.warning(f"  {cc}: no base_hate_speech_harassment.jsonl, skipping")
            continue

        # Read all records, separate pos/neg
        pos_records = []
        neg_records = []
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("is_violation"):
                    pos_records.append(d)
                else:
                    neg_records.append(d)

        pos_before = len(pos_records)
        neg_count = len(neg_records)
        target = HATE_SPEECH_POS_TARGET[cc]
        total_pos_before += pos_before
        total_before += pos_before + neg_count

        # Sample positives
        if pos_before <= target:
            sampled_pos = pos_records
            action = "kept all"
        else:
            sampled_pos = random.sample(pos_records, target)
            action = f"sampled {target}/{pos_before}"

        # Overwrite file
        all_records = sampled_pos + neg_records
        with open(fpath, "w", encoding="utf-8") as f:
            for rec in all_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        total_pos_after += len(sampled_pos)
        total_after += len(all_records)
        logger.info(f"  {cc}: {pos_before}pos/{neg_count}neg → "
                    f"{len(sampled_pos)}pos/{neg_count}neg ({action})")

    logger.info(f"  Total: {total_pos_before:,}pos → {total_pos_after:,}pos "
                f"({total_before:,} → {total_after:,} records)")
    return total_pos_before, total_pos_after


def step2_add_local_negatives():
    """Add 2K safe negative samples to each local label."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 2: Adding safe negatives to local labels")
    logger.info("=" * 60)

    total_added = 0

    for folder, cc in sorted(FOLDER_TO_CC.items()):
        csv_path = DATA_DIR / folder / "llm" / "deepseek-v4-flash" / "new_annotated_data.csv"
        if not csv_path.exists():
            logger.warning(f"  {cc}: no annotated data, skipping")
            continue

        local_labels = COUNTRY_LOCAL_LABELS.get(cc, [])
        if not local_labels:
            continue

        # Load existing texts to skip duplicates
        existing_texts = set()
        for lbl in local_labels:
            fpath = OUT_DIR / cc / f"{lbl}.jsonl"
            if fpath.exists():
                with open(fpath, encoding="utf-8") as f:
                    for line in f:
                        try:
                            d = json.loads(line.strip())
                            existing_texts.add(d.get("text", "").strip())
                        except json.JSONDecodeError:
                            continue

        # Collect safe texts from external data
        safe_texts = []
        with open(csv_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get("text", "").strip()
                if not text or text in existing_texts:
                    continue
                ext_labels = {k: _parse_float(row.get(k, 0)) for k in EXTERNAL_LABELS}
                if all(v < 0.5 for v in ext_labels.values()):
                    safe_texts.append(text)

        needed = LOCAL_NEG_TARGET
        sampled = random.sample(safe_texts, min(needed, len(safe_texts)))

        logger.info(f"  {cc}: {len(local_labels)} local labels, "
                    f"{len(safe_texts):,} safe texts available, sampling {len(sampled)}")

        # Assign to each local label
        for lbl in local_labels:
            sever = SEVERITY_MAP.get(lbl, "MEDIUM")
            records = []
            for i, text in enumerate(sampled):
                rec = {
                    "text": text,
                    "label": lbl,
                    "is_violation": False,
                    "severity": sever,
                    "detection_type": "contextual",
                    "language": "",
                    "country_code": cc,
                    "generation_strategy": "balanced_safe_negative",
                    "adversarial_technique": None,
                    "metadata": {"source": "external_safe_pool", "index": i},
                }
                records.append(rec)

            fpath = OUT_DIR / cc / f"{lbl}.jsonl"
            with open(fpath, "a", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total_added += len(records)
            logger.info(f"    {lbl}: +{len(records)}neg")

    logger.info(f"  Total added: {total_added:,} negative samples")
    return total_added


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    random.seed(42)

    p_before, p_after = step1_downsample_hate_speech()
    n_added = step2_add_local_negatives()

    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info(f"  base_hate_speech_harassment: {p_before:,}pos → {p_after:,}pos")
    logger.info(f"  local negatives added: {n_added:,}")
    logger.info("=" * 60)
