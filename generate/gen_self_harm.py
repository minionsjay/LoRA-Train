"""
Generate base_self_harm_suicide positive samples to fix P0 imbalance.

Current: 1,181 pos / 18,385 neg = 0.06:1
Target: ~3,181 pos / 18,385 neg ≈ 0.17:1 (add 2,000 pos)

Usage:
    python -m generate.gen_self_harm [--samples 2000]
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from .config import load_config
from .llm_client import LLMClient, generate_and_parse

logger = logging.getLogger(__name__)

OUT_DIR = Path("/home/ninini/Agents/Generate_data/output")

# Countries with their primary languages and cultural context
COUNTRIES = [
    {"cc": "BR", "name": "Brazil", "langs": ["Portuguese", "English"], "context": "Brazilian social media"},
    {"cc": "ID", "name": "Indonesia", "langs": ["Bahasa Indonesia", "English", "Javanese"], "context": "Indonesian social media"},
    {"cc": "MX", "name": "Mexico", "langs": ["Spanish", "English"], "context": "Mexican social media"},
    {"cc": "SA", "name": "Saudi Arabia", "langs": ["Arabic", "English"], "context": "Saudi social media"},
    {"cc": "SG", "name": "Singapore", "langs": ["English", "Chinese", "Malay", "Singlish"], "context": "Singapore social media"},
    {"cc": "TH", "name": "Thailand", "langs": ["Thai", "English"], "context": "Thai social media"},
    {"cc": "TR", "name": "Turkey", "langs": ["Turkish", "English"], "context": "Turkish social media"},
    {"cc": "ZA", "name": "South Africa", "langs": ["English", "Afrikaans", "Zulu"], "context": "South African social media"},
]

SELF_HARM_LABEL = {
    "label": "base_self_harm_suicide",
    "severity": "HIGH",
    "detection_type": "contextual",
    "description": "自残与生命安全 (Self-Harm & Suicide)。包括：自杀方法的具体描述或指导、自残行为的鼓励美化、组织或参与自杀/自残游戏（如蓝鲸游戏）、厌食症/暴食症等饮食失调的极端美化（pro-ana/pro-mia 社群）、发布自杀直播预告或告别信、怂恿他人自杀或自残。",
    "triggers": [
        "自杀/自残方法的具体描述和指导",
        "自残行为的鼓励或美化（包括'瘦身励志'的极端变体）",
        "组织或参与蓝鲸类自杀游戏",
        "厌食症/暴食症的浪漫化/美化内容（pro-ana/pro-mia）",
        "发布告别信或自杀直播预告",
        "怂恿/鼓动他人自杀或自残",
        "分享自残伤口照片或视频的描述",
        "讨论自杀工具/药物剂量的获取方式",
    ],
    "boundary": "心理健康支持论坛的求助帖、表达悲伤但无自杀意图的帖子、心理咨询师的专业建议、哀悼逝者、自杀预防热线宣传",
}

# Diverse generation strategies for self-harm content
GENERATION_STRATEGIES = [
    "social_media_post",
    "forum_comment",
    "chat_message",
    "blog_post",
    "anonymous_confession",
    "coded_language",
    "emotional_manipulation",
    "graphic_description",
    "romanticized_suffering",
    "peer_pressure",
]


def build_self_harm_prompt(country: dict, count: int, strategy_batch: list[str]) -> dict:
    """Build a prompt for generating self-harm positive samples."""
    strategies = ", ".join(strategy_batch)

    system = (
        "You are a data synthesis engine for training a multilingual content safety classifier. "
        "Your task is to generate realistic examples of content that violates the SELF-HARM / SUICIDE "
        "policy — content that encourages, glorifies, or provides guidance for suicide or self-harm. "
        "This is for a SAFETY application to protect vulnerable users. "
        "Generate DIVERSE, realistic content that reflects how real people express suicidal ideation, "
        "self-harm intent, or encouragement of self-harm across different platforms and styles. "
        "Output ONLY valid JSON objects, one per line. No markdown fences, no commentary."
    )

    user = (
        f"Generate {count} positive (violation) training samples for the self-harm / suicide detection label.\n\n"
        f"=== LABEL ===\n"
        f"Label: {SELF_HARM_LABEL['label']}\n"
        f"Severity: {SELF_HARM_LABEL['severity']}\n"
        f"Detection Type: {SELF_HARM_LABEL['detection_type']}\n"
        f"Description: {SELF_HARM_LABEL['description']}\n\n"
        f"=== TRIGGER PATTERNS (generate content matching these) ===\n"
        + "\n".join(f"- {t}" for t in SELF_HARM_LABEL['triggers']) + "\n\n"
        f"=== COUNTRY CONTEXT ===\n"
        f"Country: {country['name']}\n"
        f"Languages: {', '.join(country['langs'])} — mix naturally with code-switching\n"
        f"Cultural context: {country['context']} — reference local culture, events, platforms\n\n"
        f"=== GENERATION STRATEGIES FOR THIS BATCH ===\n"
        f"{strategies}\n\n"
        f"=== GENERATION RULES ===\n"
        f"1. Language mix: 60-70% in primary language, 30-40% English or code-switched\n"
        f"2. Vary length: 15-200 words\n"
        f"3. Vary format: social media, forum, chat, confession page, blog comment\n"
        f"4. Vary severity: some explicit, some subtle/implicit (veiled threats, coded language)\n"
        f"5. Vary demographics: teens, adults, different genders\n"
        f"6. Include both first-person (I want to...) and third-person (you should...) expressions\n"
        f"7. Reference local platforms: Twitter/X, Instagram, TikTok, Telegram, WhatsApp, local forums\n"
        f"8. IMPORTANT: The content must clearly VIOLATE the policy — it should be harmful, not help-seeking\n\n"
        f"=== BOUNDARY — DO NOT GENERATE (these are SAFE/NEGATIVE) ===\n"
        f"{SELF_HARM_LABEL['boundary']}\n\n"
        f"=== KEY DISTINCTION ===\n"
        f"VIOLATION examples: 'I'm going to kill myself tonight, here's how...', "
        f"'You should just end it all, nobody cares', 'Look at my fresh cuts, so satisfying', "
        f"'Only 45kg, need to lose 5 more — hunger is weakness leaving the body'\n"
        f"SAFE examples: 'I've been feeling really depressed lately, can anyone recommend a therapist?', "
        f"'988 Suicide Prevention Lifeline is available 24/7', "
        f"'My friend passed away last year and I miss him so much'\n\n"
        f"Output exactly {count} JSON objects, one per line:\n"
        f'{{"text": string, "label": "base_self_harm_suicide", "is_violation": true, '
        f'"severity": "HIGH", "detection_type": "contextual", '
        f'"language": string, "generation_strategy": "base_positive", '
        f'"metadata": {{"triggers_used": [string, ...], "strategy": string}}}}'
    )

    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }


async def generate_for_country(
    country: dict,
    llm_client: LLMClient,
    samples_per_country: int = 250,
    batch_size: int = 50,
    temperature: float = 0.95,
) -> list[dict]:
    """Generate self-harm positive samples for one country."""
    all_samples = []
    num_batches = (samples_per_country + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        count = min(batch_size, samples_per_country - len(all_samples))
        if count <= 0:
            break

        # Rotate strategies
        start = (batch_idx * 3) % len(GENERATION_STRATEGIES)
        strategy_batch = []
        for j in range(3):
            strategy_batch.append(GENERATION_STRATEGIES[(start + j) % len(GENERATION_STRATEGIES)])

        prompt = build_self_harm_prompt(country, count, strategy_batch)

        # Higher temperature for more diversity
        temp = temperature + (batch_idx * 0.02)
        samples = await generate_and_parse(llm_client, prompt, min(temp, 1.0))

        for s in samples:
            s.setdefault("country_code", country["cc"])
            s.setdefault("label", SELF_HARM_LABEL["label"])
            s.setdefault("is_violation", True)
            s.setdefault("severity", SELF_HARM_LABEL["severity"])
            s.setdefault("detection_type", SELF_HARM_LABEL["detection_type"])
            s.setdefault("language", country["langs"][0])
            s.setdefault("generation_strategy", "base_positive")
            s.setdefault("adversarial_technique", None)

        all_samples.extend(samples)
        logger.info(f"  {country['cc']} batch {batch_idx+1}/{num_batches}: "
                    f"got {len(samples)} samples (total: {len(all_samples)})")

        if len(samples) == 0:
            logger.warning(f"  {country['cc']} batch {batch_idx+1}: empty response, continuing...")
            continue

    return all_samples


async def generate_all(samples_per_country: int = 250, config_path: str = "config.yaml"):
    """Generate self-harm positive samples for all 8 countries."""
    config = load_config(config_path)
    llm_client = LLMClient(config.llm, config.proxy)

    total_target = samples_per_country * len(COUNTRIES)
    logger.info(f"Target: {total_target} self-harm positive samples "
                f"({samples_per_country} per country × {len(COUNTRIES)} countries)")

    try:
        grand_total = 0
        for country in COUNTRIES:
            cc = country["cc"]
            logger.info(f"\n{'='*50}")
            logger.info(f"Generating for {cc} ({country['name']})...")
            logger.info(f"{'='*50}")

            samples = await generate_for_country(
                country, llm_client,
                samples_per_country=samples_per_country,
                batch_size=50,
            )

            if samples:
                # Write to output
                out_dir = OUT_DIR / cc
                out_dir.mkdir(parents=True, exist_ok=True)
                fpath = out_dir / "base_self_harm_suicide.jsonl"

                with open(fpath, "a", encoding="utf-8") as f:
                    for s in samples:
                        f.write(json.dumps(s, ensure_ascii=False) + "\n")

                logger.info(f"  {cc}: wrote {len(samples)} samples → {fpath}")
                grand_total += len(samples)
            else:
                logger.warning(f"  {cc}: NO samples generated!")

        logger.info(f"\n{'='*50}")
        logger.info(f"GRAND TOTAL: {grand_total} samples generated (target: {total_target})")
        logger.info(f"{'='*50}")

    finally:
        await llm_client.close()

    return grand_total


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate self-harm positive samples via LLM")
    parser.add_argument("--samples", type=int, default=250,
                        help="Samples per country (default: 250, total: 2000)")
    parser.add_argument("--config", "-c", default="config.yaml",
                        help="Path to config file")
    parser.add_argument("--countries", help="Comma-separated country codes (default: all 8)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.countries:
        # Filter countries
        codes = [c.strip() for c in args.countries.split(",")]
        COUNTRIES[:] = [c for c in COUNTRIES if c["cc"] in codes]

    asyncio.run(generate_all(samples_per_country=args.samples, config_path=args.config))
