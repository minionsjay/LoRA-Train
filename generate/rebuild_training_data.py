"""
Rebuild training data into data_v2/ with new label naming and targets.

Targets:
  1. Local positives: 1,000-1,500 per label (high diversity)
  2. Base positives: 1,000+ per base label per country
  3. Safe negatives: 2,000-3,000 per country (60% easy, 40% hard)

New label names (simplified):
  Base: base_violence_gore, base_csam, base_hate_speech, base_spam_fraud, base_self_harm
  Local: {cc}_{name} e.g. sg_racial_religious_harmony, th_lese_majeste
  Safe: safe

CSV format: text,label_name,source
  source: external_integration | base_positive | positive | balanced_safe_negative |
          llm_generated | external_dataset

Usage:
  python -m generate.rebuild_training_data --stage 1
  python -m generate.rebuild_training_data --stage 2
  python -m generate.rebuild_training_data --all
"""

import asyncio
import csv
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

OUT_DIR = Path("/home/ninini/Agents/Generate_data/output")
DATA_V2_DIR = Path("/home/ninini/Agents/Generate_data/data_v2")
STAGE_FILE = DATA_V2_DIR / ".stage1_done.json"

# Old → New label name mapping
LABEL_RENAME = {
    "base_hate_speech_harassment": "base_hate_speech",
    "base_violence_dangerous_behavior": "base_violence_gore",
    "base_sexual_content_child_safety": "base_csam",
    "base_self_harm_suicide": "base_self_harm",
    "base_spam_deceptive_practices": "base_spam_fraud",
    "local_br_structural_racism": "br_structural_racism",
    "local_br_political_extremism": "br_political_extremism",
    "local_id_sara_violation": "id_sara_violation",
    "local_id_pornography_slang": "id_pornography_slang",
    "local_mx_gender_violence": "mx_gender_violence",
    "local_mx_narco_culture": "mx_narco_culture",
    "local_sa_blasphemy_anti_islam": "sa_blasphemy_anti_islam",
    "local_sa_immorality_lgbtq": "sa_immorality_lgbtq",
    "local_sa_anti_state": "sa_anti_state",
    "local_sg_racial_religious_harmony": "sg_racial_religious_harmony",
    "local_sg_vulgarity_singlish": "sg_vulgarity_singlish",
    "local_th_lese_majeste": "th_lese_majeste",
    "local_th_political_instigation": "th_political_instigation",
    "local_tr_insulting_state": "tr_insulting_state",
    "local_tr_separatism_terror": "tr_separatism_terror",
    "local_za_severe_racism": "za_severe_racism",
    "local_za_xenophobia": "za_xenophobia",
}

# New base and local label lists
NEW_BASE_LABELS = [
    "base_violence_gore", "base_csam", "base_hate_speech",
    "base_spam_fraud", "base_self_harm",
]

OLD_BASE_LABELS = [
    "base_hate_speech_harassment", "base_violence_dangerous_behavior",
    "base_sexual_content_child_safety", "base_self_harm_suicide",
    "base_spam_deceptive_practices",
]

COUNTRIES = [
    ("BR", "Brazil", ["pt", "en"]),
    ("ID", "Indonesia", ["id", "en", "jv"]),
    ("MX", "Mexico", ["es", "en"]),
    ("SA", "Saudi Arabia", ["ar", "en"]),
    ("SG", "Singapore", ["en", "zh", "ms"]),
    ("TH", "Thailand", ["th", "en"]),
    ("TR", "Turkey", ["tr", "en"]),
    ("ZA", "South Africa", ["en", "af", "zu"]),
]

COUNTRY_NAME_MAP = {c[0]: c[1] for c in COUNTRIES}

# Per-country local labels (new names)
COUNTRY_LOCAL_LABELS = {
    "BR": ["br_political_extremism", "br_structural_racism"],
    "ID": ["id_sara_violation", "id_pornography_slang"],
    "MX": ["mx_gender_violence", "mx_narco_culture"],
    "SA": ["sa_blasphemy_anti_islam", "sa_immorality_lgbtq", "sa_anti_state"],
    "SG": ["sg_racial_religious_harmony", "sg_vulgarity_singlish"],
    "TH": ["th_lese_majeste", "th_political_instigation"],
    "TR": ["tr_insulting_state", "tr_separatism_terror"],
    "ZA": ["za_severe_racism", "za_xenophobia"],
}

# Targets
LOCAL_POS_TARGET = (1000, 1500)
BASE_POS_TARGET = (1000, 2000)
SAFE_NEG_TARGET = (2000, 3000)
SAFE_EASY_RATIO = 0.6
SAFE_HARD_RATIO = 0.4

# Source tag mapping
SOURCE_MAP = {
    "external_integration": "external_integration",
    "base_positive": "llm_generated",
    "positive": "llm_generated",
    "base_general_negative": "llm_generated",
    "general_negative": "llm_generated",
    "base_boundary_negative": "llm_generated",
    "boundary_negative": "llm_generated",
    "balanced_safe_negative": "external_integration",
    "adversarial_augmentation": "llm_generated",
    "external_dataset": "external_dataset",
}


def _rename_label(old_name: str) -> str:
    """Convert old label name to new shortened name."""
    return LABEL_RENAME.get(old_name, old_name)


def _map_source(generation_strategy: str, metadata: dict | None = None) -> str:
    """Map generation strategy to source tag."""
    if metadata and metadata.get("source", "").startswith((
        "arabic_hate_speech", "turkish_hate_speech", "multilingual",
        "measuring_hate_speech", "davidson",
    )):
        return "external_dataset"
    return SOURCE_MAP.get(generation_strategy, generation_strategy)


# ===========================================================================
# Stage 1: Extract
# ===========================================================================

def stage1_extract():
    """Extract data from existing JSONL files into staging."""
    logger.info("=" * 60)
    logger.info("Stage 1: Extracting existing data")
    logger.info("=" * 60)

    DATA_V2_DIR.mkdir(parents=True, exist_ok=True)
    staging = {}

    for cc, name, langs in COUNTRIES:
        cc_dir = OUT_DIR / cc
        if not cc_dir.is_dir():
            logger.warning(f"  No data for {cc}")
            continue

        local_pos = defaultdict(list)   # new_label -> [(text, source), ...]
        base_pos = defaultdict(list)
        safe_pool = []                  # [(text, source), ...]
        hard_neg_pool = []

        for fpath in sorted(cc_dir.glob("*.jsonl")):
            old_label = fpath.stem
            new_label = _rename_label(old_label)

            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = d.get("text", "").strip()
                    if not text:
                        continue

                    is_viol = d.get("is_violation", False)
                    if isinstance(is_viol, str):
                        is_viol = is_viol.lower() == "true"
                    strat = d.get("generation_strategy", "")
                    source = _map_source(strat, d.get("metadata"))

                    if old_label.startswith("local_") and is_viol:
                        local_pos[new_label].append((text, source))
                    elif old_label in OLD_BASE_LABELS and is_viol:
                        base_pos[new_label].append((text, source))
                    elif not is_viol:
                        safe_pool.append((text, source))
                        if _is_hard_neg(text, cc):
                            hard_neg_pool.append((text, source))

        # Deduplicate
        local_pos = {lbl: _dedup_texts(items) for lbl, items in local_pos.items()}
        base_pos = {lbl: _dedup_texts(items) for lbl, items in base_pos.items()}
        safe_pool = _dedup_texts(safe_pool)
        hard_neg_pool = _dedup_texts(hard_neg_pool)

        # Downsample local pos to 1,500 max
        local_staged = {}
        for lbl, items in local_pos.items():
            if len(items) > LOCAL_POS_TARGET[1]:
                items = random.sample(items, LOCAL_POS_TARGET[1])
            local_staged[lbl] = items
            n = len(items)
            flag = f"OK ({n})" if n >= LOCAL_POS_TARGET[0] else f"NEED +{LOCAL_POS_TARGET[0]-n}"
            logger.info(f"  local {lbl}: {flag}")

        # Downsample base pos to 2,000 max
        base_staged = {}
        for lbl in NEW_BASE_LABELS:
            items = base_pos.get(lbl, [])
            if len(items) > BASE_POS_TARGET[1]:
                items = random.sample(items, BASE_POS_TARGET[1])
            base_staged[lbl] = items
            n = len(items)
            flag = f"OK ({n})" if n >= BASE_POS_TARGET[0] else f"NEED +{BASE_POS_TARGET[0]-n}"
            if n > 0:
                logger.info(f"  base {lbl}: {flag}")
            else:
                logger.info(f"  base {lbl}: MISSING - need {BASE_POS_TARGET[0]}")

        # Classify safe: easy vs hard
        easy_safe = [(t, s) for t, s in safe_pool if (t, s) not in set(hard_neg_pool)]
        easy_safe = easy_safe[:int(SAFE_NEG_TARGET[1] * SAFE_EASY_RATIO)]
        hard_safe = hard_neg_pool[:int(SAFE_NEG_TARGET[1] * SAFE_HARD_RATIO)]

        logger.info(f"  safe: {len(easy_safe)} easy + {len(hard_safe)} hard")

        staging[cc] = {
            "local_pos": local_staged,
            "base_pos": base_staged,
            "easy_safe": easy_safe,
            "hard_safe": hard_safe,
        }

    # Save staging
    staging_json = {}
    for cc, data in staging.items():
        staging_json[cc] = {
            "local_pos": {lbl: [list(it) for it in items]
                          for lbl, items in data["local_pos"].items()},
            "base_pos": {lbl: [list(it) for it in items]
                         for lbl, items in data["base_pos"].items()},
            "easy_safe": [list(it) for it in data["easy_safe"]],
            "hard_safe": [list(it) for it in data["hard_safe"]],
        }

    with open(DATA_V2_DIR / "staging.json", "w", encoding="utf-8") as f:
        json.dump(staging_json, f, ensure_ascii=False, indent=2)

    # Gap analysis
    gaps = {"base_pos": {}, "local_pos": {}, "safe_neg": {}}
    for cc, _name, _langs in COUNTRIES:
        data = staging.get(cc, {})
        if not data:
            continue
        for lbl in NEW_BASE_LABELS:
            n = len(data.get("base_pos", {}).get(lbl, []))
            if n < BASE_POS_TARGET[0]:
                gaps["base_pos"][f"{cc}/{lbl}"] = BASE_POS_TARGET[0] - n
        for lbl, items in data.get("local_pos", {}).items():
            n = len(items)
            if n < LOCAL_POS_TARGET[0]:
                gaps["local_pos"][f"{cc}/{lbl}"] = LOCAL_POS_TARGET[0] - n
        easy = len(data.get("easy_safe", []))
        hard = len(data.get("hard_safe", []))
        total = easy + hard
        if total < SAFE_NEG_TARGET[0]:
            gaps["safe_neg"][cc] = {
                "total": total,
                "need_total": SAFE_NEG_TARGET[0] - total,
                "need_hard": max(0, int(SAFE_NEG_TARGET[0] * SAFE_HARD_RATIO) - hard),
            }
        elif hard < int(total * SAFE_HARD_RATIO):
            need = int(SAFE_NEG_TARGET[1] * SAFE_HARD_RATIO) - hard
            if need > 0:
                gaps["safe_neg"][cc] = {
                    "total": total, "need_total": 0,
                    "need_hard": need,
                }

    with open(DATA_V2_DIR / "gaps.json", "w") as f:
        json.dump(gaps, f, ensure_ascii=False, indent=2)
    with open(STAGE_FILE, "w") as f:
        json.dump({"stage1_done": True, "gaps": gaps}, f)
    _print_gaps(gaps)
    return staging, gaps


def _dedup_texts(items: list) -> list:
    """Deduplicate by text, keeping first source."""
    seen = set()
    result = []
    for text, source in items:
        if text not in seen:
            seen.add(text)
            result.append((text, source))
    return result


def _is_hard_neg(text: str, cc: str) -> bool:
    """Detect hard negative candidates."""
    text_lower = text.lower()
    # Broad set of intensity/swear words
    markers = [
        "fuck", "shit", "damn", "hell", "stupid", "idiot", "crazy", "kill", "die",
        "hate", "angry", "terrible", "worst", "awful",
        "anjing", "anjir", "njir", "gila", "bangsat", "bego", "tolol", "bacot",
        "jancok", "kampret", "kontol",
        "ไอ้", "ควย", "เหี้ย", "บ้า", "สัตว์", "ถ่อย", "เชี่ย",
        "aptal", "salak", "mal", "lan", "ulan", "yavşak", "şerefsiz",
        "caralho", "porra", "merda", "burro", "idiota", "lixo",
        "chinga", "pendejo", "cabron", "mierda", "puta", "no mames",
        "غبي", "كلب", "حمار", "لعنة", "تبا",
        "傻逼", "妈的", "操", "草", "靠", "垃圾", "他妈的",
        "kak", "dom", "fok", "voetsek",
        "sia", "sian", "walao", "knn", "cb",
    ]
    return any(m in text_lower for m in markers)


def _print_gaps(gaps: dict):
    logger.info(f"\n{'='*60}")
    logger.info("GAP ANALYSIS")
    logger.info("=" * 60)
    total = 0
    if gaps.get("base_pos"):
        logger.info("\nBase positives to generate:")
        for key, need in sorted(gaps["base_pos"].items()):
            logger.info(f"  {key}: +{need}")
            total += need
    if gaps.get("local_pos"):
        logger.info("\nLocal positives to generate:")
        for key, need in sorted(gaps["local_pos"].items()):
            logger.info(f"  {key}: +{need}")
            total += need
    if gaps.get("safe_neg"):
        logger.info("\nSafe hard negatives to generate:")
        for cc, info in sorted(gaps["safe_neg"].items()):
            logger.info(f"  {cc}: +{info.get('need_hard', 0)} hard neg")
            total += info.get("need_hard", 0)
    logger.info(f"\nTOTAL LLM generation needed: ~{total} samples")


# ===========================================================================
# Stage 2: LLM Generation
# ===========================================================================

def build_base_gen_prompt(label: str, country_name: str, lang: str, count: int) -> dict:
    """Prompt for generating base label positive samples in target language."""
    label_descriptions = {
        "base_violence_gore": "暴力与血腥内容 (Violence & Gore): 煽动暴力、传播武器制作方法、极端血腥/酷刑描述、组织暴力活动",
        "base_csam": "儿童保护与色情内容 (CSAM & Sexual Content): 涉及未成年人的性化描述、未经同意的私密影像传播、公开成人色情",
        "base_hate_speech": "仇恨言论与骚扰 (Hate Speech & Harassment): 基于种族/宗教/性别等特征的攻击、系统性骚扰、非人化叙事",
        "base_spam_fraud": "欺诈与垃圾信息 (Spam & Fraud): 钓鱼链接、虚假中奖通知、庞氏骗局、虚假投资推广、批量广告",
        "base_self_harm": "自残与自杀 (Self-Harm & Suicide): 自杀方法描述、自残美化、自杀游戏、厌食症美化、怂恿他人自杀",
    }
    desc = label_descriptions.get(label, label)

    system = (
        f"You are generating training data for a multilingual content safety classifier. "
        f"Create realistic, diverse examples of content that VIOLATES the policy: {desc}. "
        f"Adapt the content to {country_name}'s language ({lang}) and cultural context. "
        f"Output ONLY valid JSON objects, one per line. No markdown fences."
    )
    user = (
        f"Generate {count} DIVERSE violation samples for label '{label}' in {lang}.\n"
        f"Country context: {country_name}. Reference local platforms, culture, current events.\n\n"
        f"Vary: length (10-150 words), format (post/comment/chat/message), severity, "
        f"writing style. Include slang and natural expressions.\n\n"
        f"Output exactly {count} JSON, one per line:\n"
        f'{{"text": string, "label": "{label}", "language": string}}'
    )
    return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}


def build_hard_neg_gen_prompt(country_name: str, lang: str, count: int, swear_words: str, topics: str) -> dict:
    system = (
        "Generate SAFE but emotionally intense social media posts. "
        "These contain swear words, anger, or borderline topics but are NOT policy violations. "
        "They help train the classifier NOT to over-flag emotional content. "
        "Output ONLY valid JSON objects, one per line."
    )
    user = (
        f"Generate {count} SAFE posts from {country_name} in {lang}.\n"
        f"Allowed (these are SAFE): swear words like {swear_words} used as emotional expression; "
        f"intense complaints about: {topics}.\n"
        f"NOT allowed: targeted harassment, racial/religious hate, violence incitement, "
        f"suicide/self-harm, CSAM, spam/scam.\n"
        f"Every sample MUST be genuinely safe despite strong language.\n\n"
        f"Output {count} JSON, one per line: {{'text': string}}"
    )
    return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}


async def stage2_generate(config_path: str = "config.yaml", batch_size: int = 40):
    if not STAGE_FILE.exists():
        logger.error("Run --stage 1 first")
        return

    with open(STAGE_FILE) as f:
        stage_data = json.load(f)
    gaps = stage_data.get("gaps", {})

    with open(DATA_V2_DIR / "staging.json", encoding="utf-8") as f:
        staging = json.load(f)

    from .config import load_config
    from .llm_client import LLMClient, generate_and_parse

    config = load_config(config_path)
    llm_client = LLMClient(config.llm, config.proxy)
    total_gen = 0

    try:
        # 1. Base label positives
        if gaps.get("base_pos"):
            logger.info(f"\n{'='*60}")
            logger.info("Generating BASE LABEL POSITIVES")
            logger.info("=" * 60)
            for key, need in sorted(gaps["base_pos"].items()):
                cc, label = key.split("/", 1)
                name = COUNTRY_NAME_MAP.get(cc, cc)
                lang = [c[2][0] for c in COUNTRIES if c[0] == cc][0]
                lang_full = {"pt": "Portuguese", "id": "Bahasa Indonesia", "es": "Spanish",
                             "ar": "Arabic", "en": "English", "th": "Thai",
                             "tr": "Turkish", "af": "Afrikaans", "zh": "Chinese",
                             "ms": "Malay"}.get(lang, lang)

                logger.info(f"\n  {key}: generating {need}...")
                generated = []
                for b in range((need + batch_size - 1) // batch_size):
                    n = min(batch_size, need - len(generated))
                    prompt = build_base_gen_prompt(label, name, lang_full, n)
                    samples = await generate_and_parse(llm_client, prompt, 0.9)
                    generated.extend(samples)
                    logger.info(f"    Batch {b+1}: +{len(samples)} (total {len(generated)})")

                if label not in staging[cc]["base_pos"]:
                    staging[cc]["base_pos"][label] = []
                for s in generated:
                    staging[cc]["base_pos"][label].append([s["text"].strip(), "llm_generated"])
                total_gen += len(generated)

        # 2. Hard negatives for safe pool
        if gaps.get("safe_neg"):
            logger.info(f"\n{'='*60}")
            logger.info("Generating HARD NEGATIVES")
            logger.info("=" * 60)
            hard_neg_info = {
                "BR": ("Portuguese", "caralho, porra, merda, burro, lixo", "football losses, São Paulo traffic, government corruption"),
                "ID": ("Bahasa Indonesia", "anjing, njir, gila, bangsat, bego, tolol", "Jakarta traffic, online game rage, Gojek order fails"),
                "MX": ("Spanish", "chinga, pendejo, cabron, mierda, no mames", "futbol rage, CDMX traffic, corrupt politicians"),
                "SA": ("Arabic", "غبي, كلب, حمار, لعنة, تبا", "Riyadh traffic, football anger, high prices, service complaints"),
                "SG": ("English+Singlish", "sia, sian, walao, knn, cb", "MRT breakdown, hawker price hike, COE/HDB prices"),
                "TH": ("Thai", "ไอ้, ควย, เหี้ย, บ้า, สัตว์, เชี่ย", "Bangkok traffic, politics anger, work complaints, lottery loss"),
                "TR": ("Turkish", "aptal, salak, mal, lan, yavşak", "football rage, economy complaints, Istanbul traffic"),
                "ZA": ("English+Afrikaans", "kak, dom, fok, voetsek, eish", "load shedding anger, crime complaints, pothole rants"),
            }

            for cc, info in sorted(gaps["safe_neg"].items()):
                need = info.get("need_hard", 0)
                if need <= 0:
                    continue
                lang, swear, topics = hard_neg_info.get(cc, hard_neg_info["SG"])
                name = COUNTRY_NAME_MAP.get(cc, cc)
                logger.info(f"\n  {cc}: generating {need} hard negatives...")
                generated = []
                for b in range((need + batch_size - 1) // batch_size):
                    n = min(batch_size, need - len(generated))
                    prompt = build_hard_neg_gen_prompt(name, lang, n, swear, topics)
                    samples = await generate_and_parse(llm_client, prompt, 0.9)
                    generated.extend(samples)
                    logger.info(f"    Batch {b+1}: +{len(samples)} (total {len(generated)})")

                staging[cc]["hard_safe"].extend(
                    [[s["text"].strip(), "llm_generated"] for s in generated]
                )
                total_gen += len(generated)

        # Save
        with open(DATA_V2_DIR / "staging.json", "w", encoding="utf-8") as f:
            json.dump(staging, f, ensure_ascii=False, indent=2)

        logger.info(f"\nStage 2 COMPLETE: {total_gen} samples generated")

    finally:
        await llm_client.close()

    return total_gen


# ===========================================================================
# Stage 3: Build CSVs
# ===========================================================================

def stage3_build():
    staging_path = DATA_V2_DIR / "staging.json"
    if not staging_path.exists():
        logger.error("No staging data. Run --stage 1 first.")
        return

    with open(staging_path, encoding="utf-8") as f:
        staging = json.load(f)

    logger.info("=" * 60)
    logger.info("Stage 3: Building final CSVs (text,label_name,source)")
    logger.info("=" * 60)

    for cc, name, langs in COUNTRIES:
        data = staging.get(cc, {})
        if not data:
            continue

        rows = []  # [(text, label_name, source), ...]

        # 1. Local positives
        for lbl, items in data.get("local_pos", {}).items():
            item_list = list(items)
            if len(item_list) > LOCAL_POS_TARGET[1]:
                item_list = random.sample(item_list, LOCAL_POS_TARGET[1])
            for text, source in item_list:
                rows.append((text.strip(), lbl, source))

        # 2. Base positives
        for lbl in NEW_BASE_LABELS:
            items = data.get("base_pos", {}).get(lbl, [])
            item_list = list(items)
            if len(item_list) > BASE_POS_TARGET[1]:
                item_list = random.sample(item_list, BASE_POS_TARGET[1])
            for text, source in item_list:
                rows.append((text.strip(), lbl, source))

        # 3. Safe negatives
        easy = list(data.get("easy_safe", []))
        hard = list(data.get("hard_safe", []))
        random.shuffle(easy)
        random.shuffle(hard)
        easy = easy[:int(SAFE_NEG_TARGET[1] * SAFE_EASY_RATIO)]
        hard = hard[:int(SAFE_NEG_TARGET[1] * SAFE_HARD_RATIO)]
        for text, source in easy + hard:
            rows.append((text.strip(), "safe", source))

        random.shuffle(rows)

        # Write CSV
        csv_path = DATA_V2_DIR / f"train_{cc}.csv"
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label_name", "source"])
            writer.writerows(rows)

        # Stats
        safe_count = sum(1 for _, lbl, _ in rows if lbl == "safe")
        base_counts = defaultdict(int)
        local_counts = defaultdict(int)
        source_counts = defaultdict(int)
        for _, lbl, src in rows:
            if lbl in NEW_BASE_LABELS:
                base_counts[lbl] += 1
            elif lbl != "safe":
                local_counts[lbl] += 1
            source_counts[src] += 1

        logger.info(f"\n{cc} ({name}): {csv_path}")
        logger.info(f"  Total: {len(rows):,} rows ({len(rows)-safe_count} pos + {safe_count} safe)")
        for lbl, cnt in sorted(local_counts.items()):
            logger.info(f"    {lbl}: {cnt}")
        for lbl, cnt in sorted(base_counts.items()):
            logger.info(f"    {lbl}: {cnt}")
        logger.info(f"    safe: {safe_count}")
        logger.info(f"  Sources: {dict(source_counts)}")

    logger.info(f"\nStage 3 COMPLETE")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2, 3])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--config", "-c", default="config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    random.seed(42)

    if args.all or args.stage == 1:
        stage1_extract()
    if args.all or args.stage == 2:
        if args.dry_run:
            with open(STAGE_FILE) as f:
                _print_gaps(json.load(f).get("gaps", {}))
        else:
            asyncio.run(stage2_generate(args.config))
    if args.all or args.stage == 3:
        stage3_build()
