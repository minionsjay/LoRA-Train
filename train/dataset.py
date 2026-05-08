import json
import re
import logging
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from .config import DataConfig

logger = logging.getLogger(__name__)

# Language field normalization map
LANG_NORMALIZE = {
    "indonesian": "id", "bahasa indonesia": "id", "bahasa": "id",
    "javanese": "jv", "bahasa jawa": "jv",
    "english": "en",
    "turkish": "tr", "türkçe": "tr",
    "portuguese": "pt", "português": "pt", "pt-br": "pt", "pt_br": "pt",
    "spanish": "es", "español": "es", "es-mx": "es",
    "arabic": "ar", "العربية": "ar",
    "thai": "th", "ภาษาไทย": "th",
    "chinese": "zh", "中文": "zh", "zh-cn": "zh", "zh-sg": "zh",
    "kurdish": "ku", "kurmancî": "ku",
    "afrikaans": "af",
    "russian": "ru", "русский": "ru",
    "mixed": "mixed",
}

# Known label typos to fix
LABEL_FIXES = {
    "local_th_political_instication": "local_th_political_instigation",
    "local_th_political_instigationn": "local_th_political_instigation",
}


def normalize_language(lang: str) -> str:
    """Normalize language field to ISO 639-1 code."""
    if not lang or lang == "?":
        return "unknown"
    lang = lang.strip().lower().split(",")[0].split("-")[0].split("_")[0].split("/")[0]
    return LANG_NORMALIZE.get(lang, lang)


def load_and_clean_jsonl(jsonl_path: str) -> list[dict]:
    """Load a JSONL file, clean language fields, fix label typos, deduplicate."""
    samples = []
    seen_texts = set()

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Fix label typos
            label = d.get("label", "")
            if label in LABEL_FIXES:
                d["label"] = LABEL_FIXES[label]

            # Normalize language
            d["language"] = normalize_language(d.get("language", ""))

            # Deduplicate by text
            text = d.get("text", "").strip()
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)

            samples.append(d)

    return samples


def build_country_label_map(samples: list[dict], country_code: str) -> tuple[list[str], dict[str, str]]:
    """Build the ordered label list and detection_type map for a country."""
    labels = set()
    label_detection_types = {}

    base_labels = [
        "base_violence_dangerous_behavior",
        "base_hate_speech_harassment",
        "base_sexual_content_child_safety",
        "base_self_harm_suicide",
        "base_spam_deceptive_practices",
    ]
    for bl in base_labels:
        labels.add(bl)

    for s in samples:
        label = s.get("label", "")
        # Only accept labels that belong to this country (base or local)
        is_base = label in base_labels
        is_local = label.startswith(f"local_{country_code.lower()}_")
        if is_base or is_local:
            labels.add(label)
            if label not in label_detection_types:
                dt = s.get("detection_type", "contextual")
                label_detection_types[label] = dt

    # Order: base labels first, then local labels sorted
    ordered = [l for l in base_labels] + sorted(
        [l for l in labels if l not in base_labels]
    )

    return ordered, label_detection_types


class SafetyDataset(Dataset):
    def __init__(
        self,
        samples: list[dict],
        country_code: str,
        label_list: list[str],
        tokenizer: AutoTokenizer,
        max_length: int = 256,
    ):
        self.country_code = country_code
        self.label_list = label_list
        self.label_to_idx = {l: i for i, l in enumerate(label_list)}
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_labels = len(label_list)

        self.texts = []
        self.labels = []  # multi-hot tensors

        for s in samples:
            text = s.get("text", "")
            if not text:
                continue

            label_name = s.get("label", "")
            is_violation = s.get("is_violation", False)

            # Build multi-hot vector
            label_vec = [0.0] * self.num_labels

            if label_name in self.label_to_idx:
                idx = self.label_to_idx[label_name]
                label_vec[idx] = 1.0 if is_violation else 0.0

            self.texts.append(text)
            self.labels.append(label_vec)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label_vec = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_vec, dtype=torch.float),
        }


def _load_csv_samples(csv_path: str, country_code: str) -> list[dict]:
    """Load samples from a CSV file, filtering by country_code."""
    import csv
    samples = []
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("country_code", "").upper() == country_code.upper():
                # Convert is_violation string to bool
                row["is_violation"] = row.get("is_violation", "false").lower() == "true"
                samples.append(row)
    return samples


def prepare_country_data(
    data_config: DataConfig,
    country_code: str,
    tokenizer: AutoTokenizer,
) -> tuple[SafetyDataset, SafetyDataset, SafetyDataset, list[str], dict[str, str]]:
    """Load, clean, split, and create datasets for one country.

    Supports two data sources:
    1. Per-country JSONL directory: data_config.input_dir / country_code / *.jsonl
    2. Merged CSV file: data_config.input_dir / safety_training_data.csv
    """
    input_dir = Path(data_config.input_dir)

    # Try merged CSV first, then per-country JSONL directory
    merged_csv = input_dir / "safety_training_data.csv"
    country_dir = input_dir / country_code

    all_samples = []

    if merged_csv.exists():
        logger.info(f"Loading from merged CSV: {merged_csv}")
        all_samples = _load_csv_samples(str(merged_csv), country_code)
        logger.info(f"Loaded {len(all_samples)} samples for {country_code}")
    elif country_dir.is_dir():
        logger.info(f"Loading from JSONL directory: {country_dir}")
        for fpath in sorted(country_dir.glob("*.jsonl")):
            samples = load_and_clean_jsonl(str(fpath))
            logger.info(f"Loaded {len(samples)} clean samples from {fpath.name}")
            all_samples.extend(samples)
    else:
        raise FileNotFoundError(
            f"Data not found. Looked for:\n"
            f"  - CSV: {merged_csv}\n"
            f"  - JSONL: {country_dir}/\n"
            f"Copy data from the source machine or run 'python -m generate.data_mgmt export-csv' first."
        )

    if not all_samples:
        raise ValueError(f"No samples found for country {country_code}")

    # Build label map
    label_list, label_detection_types = build_country_label_map(all_samples, country_code)
    logger.info(f"Labels for {country_code}: {label_list}")

    # Filter out samples whose labels are not in the country's label list
    valid_label_set = set(label_list)
    filtered = [s for s in all_samples if s.get("label") in valid_label_set]

    # Stratified split by label
    labels_for_split = [s["label"] for s in filtered]
    train_val, test = train_test_split(
        filtered,
        test_size=data_config.test_ratio,
        stratify=labels_for_split,
        random_state=42,
    )

    train_val_labels = [s["label"] for s in train_val]
    val_ratio_adjusted = data_config.val_ratio / (data_config.train_ratio + data_config.val_ratio)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio_adjusted,
        stratify=train_val_labels,
        random_state=42,
    )

    logger.info(f"Split: train={len(train)} val={len(val)} test={len(test)}")

    # Create datasets
    train_ds = SafetyDataset(train, country_code, label_list, tokenizer, max_length=256)
    val_ds = SafetyDataset(val, country_code, label_list, tokenizer, max_length=256)
    test_ds = SafetyDataset(test, country_code, label_list, tokenizer, max_length=256)

    return train_ds, val_ds, test_ds, label_list, label_detection_types
