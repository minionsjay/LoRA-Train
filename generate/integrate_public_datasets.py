"""
Integrate public Hugging Face datasets into the training data.

Datasets integrated:
1. tdavidson/hate_speech_offensive (25K)       -> base labels (English)
2. ucberkeley-dlab/measuring-hate-speech (136K) -> base labels (English)
3. arbml/Arabic_Hate_Speech (10K)              -> SA local labels + base
4. FrancophonIA/multilingual-hatespeech (279K)  -> multi-country base
5. manueltonneau/arabic-hate-speech-superset (449K) -> SA (requires HF auth)
6. manueltonneau/turkish-hate-speech-superset (41K) -> TR (requires HF auth)

Note: ctoraman/large-scale-hate-speech-turkish-v2 (60K) is skipped because
it only contains Tweet IDs, not actual text.

Usage:
    python -m generate.integrate_public_datasets --all
    python -m generate.integrate_public_datasets --dataset davidson
    python -m generate.integrate_public_datasets --dry-run
"""

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

OUT_DIR = Path("/home/ninini/Agents/Generate_data/output")

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

BASE_LABELS = [
    "base_violence_dangerous_behavior",
    "base_hate_speech_harassment",
    "base_sexual_content_child_safety",
    "base_self_harm_suicide",
    "base_spam_deceptive_practices",
]


def _make_record(text, label, is_violation, cc, generation_strategy, language="", metadata=None):
    """Build a JSON record matching the existing format."""
    meta = LABEL_META.get(label, {"severity": "MEDIUM", "detection_type": "contextual"})
    return {
        "text": text.strip(),
        "label": label,
        "is_violation": is_violation,
        "severity": meta["severity"],
        "detection_type": meta["detection_type"],
        "language": language,
        "country_code": cc,
        "generation_strategy": generation_strategy,
        "adversarial_technique": None,
        "metadata": metadata or {},
    }


def load_existing_texts() -> dict[str, set]:
    """Load all existing texts per country for dedup. Returns {cc: set of texts}."""
    existing = defaultdict(set)
    if not OUT_DIR.exists():
        return existing
    for cc_dir in OUT_DIR.iterdir():
        if not cc_dir.is_dir():
            continue
        for fpath in cc_dir.glob("*.jsonl"):
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        d = json.loads(line)
                        existing[cc_dir.name].add(d.get("text", "").strip())
                    except json.JSONDecodeError:
                        continue
    return existing


def _write_records(records_by_label_cc: dict) -> int:
    """Write records to output JSONL files, keyed by (cc, label)."""
    written = 0
    for (cc, label), records in records_by_label_cc.items():
        cc_dir = OUT_DIR / cc
        cc_dir.mkdir(parents=True, exist_ok=True)
        fpath = cc_dir / f"{label}.jsonl"
        with open(fpath, "a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        written += len(records)
        pos = sum(1 for r in records if r["is_violation"])
        logger.info(f"  {cc}/{label}: +{pos}pos +{len(records) - pos}neg")
    return written


def _dedup_check(existing_texts, cc, text):
    """Check if text already exists in target country. Returns True if duplicate."""
    if text in existing_texts.get(cc, set()):
        return True
    existing_texts[cc].add(text)
    return False


# ===========================================================================
# Dataset 1: Davidson Hate Speech & Offensive Language (25K)
# ===========================================================================

def integrate_davidson(existing_texts, max_pos=3000, max_neg=3000):
    """Davidson Hate Speech (25K English) -> base labels across countries."""
    logger.info("=" * 60)
    logger.info("Dataset 1: Davidson Hate Speech & Offensive (25K)")
    logger.info("=" * 60)

    from datasets import load_dataset
    ds = load_dataset("tdavidson/hate_speech_offensive", split="train")
    logger.info(f"  Loaded {len(ds):,} rows")

    target_ccs = ["BR", "ID", "MX", "SG", "TH", "ZA"]
    records = defaultdict(list)
    stats = {"pos": 0, "neg": 0, "dup": 0}

    # Hate speech (class=0) and offensive (class=1) -> base_hate_speech_harassment positive
    for row in ds:
        if stats["pos"] >= max_pos:
            break
        if int(row["class"]) not in (0, 1):
            continue
        text = row.get("tweet", "").strip()
        if not text:
            continue
        cc = random.choice(target_ccs)
        if _dedup_check(existing_texts, cc, text):
            stats["dup"] += 1
            continue
        records[(cc, "base_hate_speech_harassment")].append(
            _make_record(text, "base_hate_speech_harassment", True, cc,
                         "external_integration", "en",
                         {"source": "davidson_hate_speech",
                          "class": int(row["class"])}))
        stats["pos"] += 1

    # Neither (class=2) -> safe negatives
    for row in ds:
        if stats["neg"] >= max_neg:
            break
        if int(row["class"]) != 2:
            continue
        text = row.get("tweet", "").strip()
        if not text:
            continue
        cc = random.choice(target_ccs)
        if _dedup_check(existing_texts, cc, text):
            stats["dup"] += 1
            continue
        label = random.choice(BASE_LABELS)
        records[(cc, label)].append(
            _make_record(text, label, False, cc, "external_integration", "en",
                         {"source": "davidson_hate_speech"}))
        stats["neg"] += 1

    written = _write_records(records)
    logger.info(f"  Summary: +{stats['pos']}pos +{stats['neg']}neg, {stats['dup']}dup = {written} written")
    return written


# ===========================================================================
# Dataset 2: Measuring Hate Speech (136K English, continuous IRT scores)
# ===========================================================================

def integrate_measuring_hate(existing_texts, max_pos=8000, max_neg=8000):
    """Measuring Hate Speech (136K) -> base labels across countries."""
    logger.info("=" * 60)
    logger.info("Dataset 2: Measuring Hate Speech (136K)")
    logger.info("=" * 60)

    from datasets import load_dataset
    ds = load_dataset("ucberkeley-dlab/measuring-hate-speech", split="train")
    logger.info(f"  Loaded {len(ds):,} rows")

    # hate_speech_score > 1.0 = clear hate, < -1.0 = clear non-hate
    # Use IRT-adjusted scores for high-confidence samples only
    target_ccs = ["BR", "ID", "MX", "SG", "TH", "ZA"]
    records = defaultdict(list)
    stats = {"pos": 0, "neg": 0, "dup": 0}

    # Hate samples (high confidence: score > 1.5)
    for row in ds:
        if stats["pos"] >= max_pos:
            break
        score = row.get("hate_speech_score", 0)
        if score is None or score <= 1.5:
            continue
        text = row.get("text", "").strip()
        if not text:
            continue
        cc = random.choice(target_ccs)
        if _dedup_check(existing_texts, cc, text):
            stats["dup"] += 1
            continue
        records[(cc, "base_hate_speech_harassment")].append(
            _make_record(text, "base_hate_speech_harassment", True, cc,
                         "external_integration", "en",
                         {"source": "measuring_hate_speech",
                          "hate_score": round(float(score), 2)}))
        stats["pos"] += 1

    # Non-hate samples (high confidence: score < -1.5)
    for row in ds:
        if stats["neg"] >= max_neg:
            break
        score = row.get("hate_speech_score", 0)
        if score is None or score >= -1.5:
            continue
        text = row.get("text", "").strip()
        if not text:
            continue
        cc = random.choice(target_ccs)
        if _dedup_check(existing_texts, cc, text):
            stats["dup"] += 1
            continue
        label = random.choice(BASE_LABELS)
        records[(cc, label)].append(
            _make_record(text, label, False, cc, "external_integration", "en",
                         {"source": "measuring_hate_speech",
                          "hate_score": round(float(score), 2)}))
        stats["neg"] += 1

    written = _write_records(records)
    logger.info(f"  Summary: +{stats['pos']}pos +{stats['neg']}neg, {stats['dup']}dup = {written} written")
    return written


# ===========================================================================
# Dataset 3: Arabic Hate Speech (10K, multi-label)
# ===========================================================================

def integrate_arabic_10k(existing_texts, max_pos=3000, max_neg=3000):
    """Arabic Hate Speech (10K) -> SA labels + base."""
    logger.info("=" * 60)
    logger.info("Dataset 3: Arabic Hate Speech (10K)")
    logger.info("=" * 60)

    from datasets import load_dataset
    train_ds = load_dataset("arbml/Arabic_Hate_Speech", split="train")
    val_ds = load_dataset("arbml/Arabic_Hate_Speech", split="validation")
    logger.info(f"  Loaded {len(train_ds):,} train + {len(val_ds):,} val rows")

    records = defaultdict(list)
    stats = {"pos": 0, "neg": 0, "dup": 0}

    # Labels: is_off (offensive), is_hate (hate type), is_vlg (vulgar), is_vio (violence)
    # Map: is_hate -> local_sa_blasphemy_anti_islam / base_hate_speech_harassment
    #       is_vlg -> local_sa_immorality_lgbtq
    #       is_vio -> base_violence_dangerous_behavior
    #       is_off -> base_hate_speech_harassment

    all_rows = list(train_ds) + list(val_ds)

    for row in all_rows:
        text = row.get("tweet", "").strip()
        if not text:
            continue
        if _dedup_check(existing_texts, "SA", text):
            stats["dup"] += 1
            continue

        assigned = False

        # Hate speech -> blasphemy or base hate
        if row.get("is_hate", "NOT_HS") != "NOT_HS":
            if stats["pos"] < max_pos:
                label = ("local_sa_blasphemy_anti_islam"
                         if random.random() < 0.6
                         else "base_hate_speech_harassment")
                records[("SA", label)].append(
                    _make_record(text, label, True, "SA", "external_integration",
                                 "ar", {"source": "arabic_hate_speech_10k",
                                        "hate_type": row.get("is_hate", "")}))
                stats["pos"] += 1
                assigned = True

        # Vulgar language -> immorality/LGBTQ
        if row.get("is_vlg", "NOT_VLG") == "VLG" and not assigned:
            if stats["pos"] < max_pos:
                label = "local_sa_immorality_lgbtq"
                records[("SA", label)].append(
                    _make_record(text, label, True, "SA", "external_integration",
                                 "ar", {"source": "arabic_hate_speech_10k"}))
                stats["pos"] += 1
                assigned = True

        # Violence -> base violence
        if row.get("is_vio", "NOT_VIO") == "VIO" and not assigned:
            if stats["pos"] < max_pos // 2:
                label = "base_violence_dangerous_behavior"
                records[("SA", label)].append(
                    _make_record(text, label, True, "SA", "external_integration",
                                 "ar", {"source": "arabic_hate_speech_10k"}))
                stats["pos"] += 1
                assigned = True

        # General offensive
        if row.get("is_off", "NOT_OFF") == "OFF" and not assigned:
            if stats["pos"] < max_pos:
                label = "base_hate_speech_harassment"
                records[("SA", label)].append(
                    _make_record(text, label, True, "SA", "external_integration",
                                 "ar", {"source": "arabic_hate_speech_10k"}))
                stats["pos"] += 1
                assigned = True

        # Non-offensive -> safe negatives for SA labels
        if not assigned and row.get("is_off", "OFF") == "NOT_OFF":
            if stats["neg"] < max_neg:
                label = random.choice([
                    "local_sa_blasphemy_anti_islam",
                    "local_sa_anti_state",
                    "local_sa_immorality_lgbtq",
                ])
                records[("SA", label)].append(
                    _make_record(text, label, False, "SA", "external_integration",
                                 "ar", {"source": "arabic_hate_speech_10k"}))
                stats["neg"] += 1

    written = _write_records(records)
    logger.info(f"  Summary: +{stats['pos']}pos +{stats['neg']}neg, {stats['dup']}dup = {written} written")
    return written


# ===========================================================================
# Dataset 4: Multilingual Hate Speech (279K, 18 languages)
# ===========================================================================

def integrate_multilingual(existing_texts, max_pos_per_lang=500, max_neg_per_lang=1500):
    """Multilingual Hate Speech -> per-language base labels."""
    logger.info("=" * 60)
    logger.info("Dataset 4: Multilingual Hate Speech (279K)")
    logger.info("=" * 60)

    from datasets import load_dataset, get_dataset_config_names

    try:
        configs = get_dataset_config_names("FrancophonIA/multilingual-hatespeech-dataset")
    except Exception as e:
        logger.error(f"  Failed to get configs: {e}")
        return 0

    logger.info(f"  Available configs: {configs}")

    # Map config names to country codes
    # Known configs from the dataset: Arabic_test, Chinese_test, English_test, ...
    # Some have alternate names like Porto_test for Portuguese, Spain_test for Spanish
    config_to_cc = {}

    for cfg in configs:
        cfg_lower = cfg.lower()
        if "arabic" in cfg_lower:
            config_to_cc[cfg] = "SA"
        elif "turkish" in cfg_lower:
            config_to_cc[cfg] = "TR"
        elif "indonesian" in cfg_lower:
            config_to_cc[cfg] = "ID"
        elif "porto" in cfg_lower or "portuguese" in cfg_lower:
            config_to_cc[cfg] = "BR"
        elif "spain" in cfg_lower or "spanish" in cfg_lower:
            config_to_cc[cfg] = "MX"

    if not config_to_cc:
        logger.warning("  No matching language configs found. Skipping.")
        return 0

    logger.info(f"  Mapped to countries: {config_to_cc}")

    records = defaultdict(list)
    stats = {"pos": 0, "neg": 0, "dup": 0}

    for config_name, cc in config_to_cc.items():
        try:
            test_ds = load_dataset(
                "FrancophonIA/multilingual-hatespeech-dataset",
                config_name, split="test")
        except Exception as e:
            logger.warning(f"  Could not load {config_name}: {e}")
            continue

        logger.info(f"  {config_name} ({cc}): {len(test_ds)} rows")

        lang_pos = lang_neg = 0
        for row in test_ds:
            text = row.get("text", "").strip()
            if not text:
                continue
            if _dedup_check(existing_texts, cc, text):
                stats["dup"] += 1
                continue

            if row.get("label", 0) == 1:  # Hate
                if lang_pos >= max_pos_per_lang:
                    continue
                records[(cc, "base_hate_speech_harassment")].append(
                    _make_record(text, "base_hate_speech_harassment", True, cc,
                                 "external_integration", cc.lower(),
                                 {"source": "multilingual_hatespeech",
                                  "config": config_name}))
                lang_pos += 1
                stats["pos"] += 1
            else:  # Non-hate -> negative for a random base label
                if lang_neg >= max_neg_per_lang:
                    continue
                label = random.choice(BASE_LABELS)
                records[(cc, label)].append(
                    _make_record(text, label, False, cc, "external_integration",
                                 cc.lower(),
                                 {"source": "multilingual_hatespeech",
                                  "config": config_name}))
                lang_neg += 1
                stats["neg"] += 1

        logger.info(f"    +{lang_pos}pos +{lang_neg}neg")

    written = _write_records(records)
    logger.info(f"  Summary: +{stats['pos']}pos +{stats['neg']}neg, {stats['dup']}dup = {written} written")
    return written


# ===========================================================================
# Dataset 5: Arabic Hate Speech Superset (449K) — requires HF login
# ===========================================================================

def integrate_arabic_superset(existing_texts, max_pos=4000, max_neg=6000):
    """Arabic Hate Speech Superset (449K) -> SA labels. Requires HF auth."""
    logger.info("=" * 60)
    logger.info("Dataset 5: Arabic Hate Speech Superset (449K)")
    logger.info("=" * 60)

    from datasets import load_dataset

    try:
        ds = load_dataset("manueltonneau/arabic-hate-speech-superset", split="train")
    except Exception as e:
        logger.error(f"  Failed to load: {e}")
        logger.info("  ACTION REQUIRED:")
        logger.info("    1. Run: huggingface-cli login")
        logger.info("    2. Visit: https://huggingface.co/datasets/manueltonneau/arabic-hate-speech-superset")
        logger.info("    3. Accept the terms of use")
        logger.info("    4. Re-run this script")
        return 0

    logger.info(f"  Loaded {len(ds):,} rows")

    # Use Saudi-specific subset if available (dataset column = "saudi")
    if "dataset" in ds.features:
        saudi_ds = ds.filter(lambda x: x.get("dataset") == "saudi")
        if len(saudi_ds) > 0:
            ds = saudi_ds
            logger.info(f"  Filtered to Saudi subset: {len(ds):,} rows")

    # Separate hate vs non-hate
    hate_ds = ds.filter(lambda x: x["labels"] == 1) if len(ds) > 0 else ds
    safe_ds = ds.filter(lambda x: x["labels"] == 0) if len(ds) > 0 else ds
    logger.info(f"  Hate: {len(hate_ds):,}, Safe: {len(safe_ds):,}")

    records = defaultdict(list)
    stats = {"pos": 0, "neg": 0, "dup": 0}

    sa_labels_pos = [
        ("local_sa_blasphemy_anti_islam", 0.50),
        ("local_sa_anti_state", 0.30),
        ("base_hate_speech_harassment", 0.20),
    ]

    # Positive: hate samples
    for row in hate_ds:
        if stats["pos"] >= max_pos:
            break
        text = row.get("text", "").strip()
        if not text:
            continue
        if _dedup_check(existing_texts, "SA", text):
            stats["dup"] += 1
            continue
        label = random.choices([l[0] for l in sa_labels_pos],
                               weights=[l[1] for l in sa_labels_pos])[0]
        records[("SA", label)].append(
            _make_record(text, label, True, "SA", "external_integration", "ar",
                         {"source": "arabic_hate_speech_superset",
                          "original_dataset": row.get("dataset", "unknown")}))
        stats["pos"] += 1

    # Negative: safe Arabic texts for SA labels
    sa_labels_neg = [
        "local_sa_blasphemy_anti_islam",
        "local_sa_anti_state",
        "local_sa_immorality_lgbtq",
    ]
    for row in safe_ds:
        if stats["neg"] >= max_neg:
            break
        text = row.get("text", "").strip()
        if not text:
            continue
        if _dedup_check(existing_texts, "SA", text):
            stats["dup"] += 1
            continue
        label = random.choice(sa_labels_neg)
        records[("SA", label)].append(
            _make_record(text, label, False, "SA", "external_integration", "ar",
                         {"source": "arabic_hate_speech_superset"}))
        stats["neg"] += 1

    written = _write_records(records)
    logger.info(f"  Summary: +{stats['pos']}pos +{stats['neg']}neg, {stats['dup']}dup = {written} written")
    return written


# ===========================================================================
# Dataset 6: Turkish Hate Speech Superset (41K) — requires HF login
# ===========================================================================

def integrate_turkish_superset(existing_texts, max_pos=4000, max_neg=6000):
    """Turkish Hate Speech Superset (41K) -> TR labels. Requires HF auth."""
    logger.info("=" * 60)
    logger.info("Dataset 6: Turkish Hate Speech Superset (41K)")
    logger.info("=" * 60)

    from datasets import load_dataset

    try:
        ds = load_dataset("manueltonneau/turkish-hate-speech-superset", split="train")
    except Exception as e:
        logger.error(f"  Failed to load: {e}")
        logger.info("  ACTION REQUIRED:")
        logger.info("    1. Run: huggingface-cli login")
        logger.info("    2. Visit: https://huggingface.co/datasets/manueltonneau/turkish-hate-speech-superset")
        logger.info("    3. Accept the terms of use")
        logger.info("    4. Re-run this script")
        return 0

    logger.info(f"  Loaded {len(ds):,} rows")

    records = defaultdict(list)
    stats = {"pos": 0, "neg": 0, "dup": 0}

    tr_labels_pos = [
        ("local_tr_insulting_state", 0.40),
        ("local_tr_separatism_terror", 0.30),
        ("base_hate_speech_harassment", 0.30),
    ]

    # Positive: hate samples (labels=1)
    for row in ds:
        if stats["pos"] >= max_pos:
            break
        if row.get("labels", 0) != 1:
            continue
        text = row.get("text", "").strip()
        if not text:
            continue
        if _dedup_check(existing_texts, "TR", text):
            stats["dup"] += 1
            continue
        label = random.choices([l[0] for l in tr_labels_pos],
                               weights=[l[1] for l in tr_labels_pos])[0]
        records[("TR", label)].append(
            _make_record(text, label, True, "TR", "external_integration", "tr",
                         {"source": "turkish_hate_speech_superset",
                          "original_dataset": row.get("dataset", "unknown")}))
        stats["pos"] += 1

    # Negative: safe Turkish texts
    tr_labels_neg = [
        "local_tr_insulting_state",
        "local_tr_separatism_terror",
    ]
    for row in ds:
        if stats["neg"] >= max_neg:
            break
        if row.get("labels", 0) != 0:
            continue
        text = row.get("text", "").strip()
        if not text:
            continue
        if _dedup_check(existing_texts, "TR", text):
            stats["dup"] += 1
            continue
        label = random.choice(tr_labels_neg)
        records[("TR", label)].append(
            _make_record(text, label, False, "TR", "external_integration", "tr",
                         {"source": "turkish_hate_speech_superset"}))
        stats["neg"] += 1

    written = _write_records(records)
    logger.info(f"  Summary: +{stats['pos']}pos +{stats['neg']}neg, {stats['dup']}dup = {written} written")
    return written


# ===========================================================================
# Turkish v2 (skipped — Tweet IDs only, no text)
# ===========================================================================

def integrate_turkish_hate_v2(existing_texts):
    """SKIPPED: Turkish Hate Speech v2 only has Tweet IDs, not text."""
    logger.info("=" * 60)
    logger.info("Dataset X: Turkish Hate Speech v2 (60K) — SKIPPED")
    logger.info("=" * 60)
    logger.warning("  This dataset only contains Tweet IDs (no actual text).")
    logger.warning("  Text retrieval requires Twitter API hydration (no longer available).")
    logger.warning("  Use --dataset turkish_superset instead (requires HF login).")
    return 0


# ===========================================================================
# Registry and runner
# ===========================================================================

DATASETS = {
    "davidson": integrate_davidson,
    "measuring_hate": integrate_measuring_hate,
    "arabic_10k": integrate_arabic_10k,
    "multilingual": integrate_multilingual,
    "arabic_superset": integrate_arabic_superset,
    "turkish_superset": integrate_turkish_superset,
    "turkish_hate_v2": integrate_turkish_hate_v2,
}


def integrate_all(dry_run=False):
    """Integrate all available public datasets."""
    existing_texts = load_existing_texts()
    total = sum(len(v) for v in existing_texts.values())
    logger.info(f"Existing texts: {total:,} across {len(existing_texts)} countries")

    if dry_run:
        logger.info("DRY RUN — no files will be written")
        return

    grand = 0
    for name, func in DATASETS.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"DATASET: {name}")
        logger.info(f"{'=' * 60}")
        try:
            written = func(existing_texts)
            grand += written
            logger.info(f"  -> {written} records written")
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    logger.info(f"\n{'=' * 60}")
    logger.info(f"GRAND TOTAL: {grand:,} records integrated")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrate public Hugging Face datasets")
    parser.add_argument("--all", action="store_true", help="Integrate all datasets")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), help="Integrate a specific dataset")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    random.seed(42)

    if args.list:
        print("Available datasets:")
        for name in DATASETS:
            print(f"  - {name}")
        sys.exit(0)

    if args.dry_run:
        integrate_all(dry_run=True)
    elif args.all:
        integrate_all()
    elif args.dataset:
        existing_texts = load_existing_texts()
        total = sum(len(v) for v in existing_texts.values())
        logger.info(f"Existing texts: {total:,}")
        written = DATASETS[args.dataset](existing_texts)
        logger.info(f"Done: {written} records written")
    else:
        parser.print_help()
