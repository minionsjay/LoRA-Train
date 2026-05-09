"""
Generate Thailand-specific local label positive samples.

Current gaps:
- local_th_lese_majeste: 202 pos / 2,111 neg = 0.10:1 (need ~800 more)
- local_th_political_instigation: 832 pos / 2,095 neg = 0.40:1 (need ~400 more)

Usage:
    python -m generate.gen_th_labels [--samples-lese-majeste 800]
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

from .config import load_config
from .llm_client import LLMClient, generate_and_parse

logger = logging.getLogger(__name__)

OUT_DIR = Path("/home/ninini/Agents/Generate_data/output")

TH_CONFIG = {"cc": "TH", "name": "Thailand", "langs": ["Thai", "English"]}

LESE_MAJESTE = {
    "label": "local_th_lese_majeste",
    "severity": "CRITICAL",
    "detection_type": "contextual",
    "description": (
        "Lèse-majesté (หมิ่นพระบรมเดชานุภาพ) — 对泰国王室的不敬言论。"
        "包括：对国王/王后/王室成员的侮辱、批评王室政治影响力的言论、"
        "转发或传播反王室内容、使用隐晦代号或隐喻指代王室成员进行嘲讽、"
        "讨论王室负面新闻或丑闻、将王室成员进行不当比较或P图恶搞。"
        "注意：合法学术讨论泰国政治制度、报道已被官方公开的王室新闻、一般性讨论君主制不构成违规。"
    ),
    "triggers": [
        "直接侮辱或谩骂国王/王后/王室成员",
        "使用隐晦代号（如'十世'、'某王'等贬义用法）进行嘲讽",
        "传播未经官方确认的王室负面消息",
        "将王室成员P图恶搞或不当比较",
        "质疑王室政治中立性（带有攻击性语气的）",
        "转发反王室组织的宣传内容",
        "讨论王室财产/经费使用的攻击性言论",
    ],
    "boundary": [
        "学术性讨论泰国政治制度（不针对具体王室成员）",
        "报道已由官方媒体正式发布的王室新闻",
        "讨论君主立宪制的历史和发展（不涉及对现任王室的攻击）",
        "一般性讨论泰国历史（涉及历史人物在大众认知范围内）",
    ],
}

POLITICAL_INSTIGATION = {
    "label": "local_th_political_instigation",
    "severity": "HIGH",
    "detection_type": "contextual",
    "description": (
        "政治煽动 (Political Instigation) — 泰国语境下的政治动荡和群体对立煽动。"
        "包括：煽动抗议或集会暴力冲突、散布针对对立政治阵营的谣言、"
        "煽动对黄衫军/红杉军等历史对立阵营的仇恨、利用政治议题煽动地域对立、"
        "煽动对政府/军方/反对派的暴力对抗。"
    ),
    "triggers": [
        "煽动抗议者进行暴力冲突",
        "散布针对政治对立阵营的虚假消息以煽动仇恨",
        "号召针对特定政治人物的线下攻击",
        "利用历史政治事件（如2010年事件）煽动报复",
        "散布关于选举/投票的阴谋论以煽动对立",
        "煽动对政府/军方实施暴力抵抗",
    ],
    "boundary": [
        "合法的政治讨论和政策辩论",
        "和平集会的号召（不煽动暴力的）",
        "对政策的理性批评（不针对个人的攻击）",
        "新闻报道政治事件",
    ],
}


def build_prompt(label_config: dict, count: int, batch_theme: str) -> dict:
    system = (
        "You are a data synthesis engine for training a content safety classifier "
        "specializing in Thailand-specific cultural violations. "
        "Generate realistic examples of content that violates a specific Thai content policy. "
        "The content must reflect authentic Thai online discourse — use Thai language primarily, "
        "with natural code-switching to English where real Thai netizens would. "
        "Reference real Thai social media platforms (Pantip, Twitter/X, Facebook, Line, TikTok). "
        "Output ONLY valid JSON objects, one per line. No markdown fences, no commentary."
    )

    user = (
        f"Generate {count} positive (violation) training samples for the following Thai content policy label.\n\n"
        f"=== LABEL ===\n"
        f"Label: {label_config['label']}\n"
        f"Severity: {label_config['severity']}\n"
        f"Detection Type: {label_config['detection_type']}\n"
        f"Description: {label_config['description']}\n\n"
        f"=== TRIGGER PATTERNS ===\n"
        + "\n".join(f"- {t}" for t in label_config['triggers']) + "\n\n"
        f"=== BATCH THEME (focus on this style) ===\n"
        f"{batch_theme}\n\n"
        f"=== GENERATION RULES ===\n"
        f"1. 70-80% in Thai language, 20-30% English or code-switched\n"
        f"2. Reference Thai cultural context: locations (Bangkok, Chiang Mai), events, people\n"
        f"3. Vary length: 15-200 words\n"
        f"4. Vary format: social media post, forum comment (Pantip), Line chat, tweet, TikTok caption\n"
        f"5. Vary severity: explicit attacks, subtle sarcasm, coded references\n"
        f"6. Use authentic Thai netizen speech patterns: particles (ครับ/ค่ะ/จ้า/นะ), emojis\n"
        f"7. Include Thai slang and abbreviations where natural\n"
        f"8. IMPORTANT: Content must clearly VIOLATE the policy, not be borderline\n\n"
        f"=== BOUNDARY — DO NOT GENERATE ===\n"
        + "\n".join(f"- {b}" for b in label_config['boundary']) + "\n\n"
        f"Output exactly {count} JSON objects, one per line:\n"
        f'{{"text": string, "label": "{label_config["label"]}", "is_violation": true, '
        f'"severity": "{label_config["severity"]}", "detection_type": "{label_config["detection_type"]}", '
        f'"language": string, "generation_strategy": "positive", '
        f'"metadata": {{"triggers_used": [string, ...], "theme": string}}}}'
    )

    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }


LESE_MAJESTE_THEMES = [
    "Social media posts criticizing royal family members directly using nicknames and coded references",
    "Pantip forum threads discussing controversial royal news with sarcastic comments",
    "Twitter/X posts sharing anti-monarchy hashtags and retweeting exiled dissident content",
    "Line chat messages mocking royal ceremonies or royal projects with humor and insults",
    "TikTok comments making fun of royal portraits or comparing royals to fictional villains",
    "Facebook posts questioning royal budget allocation with inflammatory language",
    "Online discussions comparing Thailand's monarchy unfavorably to other constitutional monarchies",
    "Memes and image caption-style posts that ridicule royal family members",
]

POLITICAL_THEMES = [
    "Posts inciting violence at political protests/rallies in Bangkok",
    "Spreading conspiracy theories about election fraud to stir division",
    "Calls for violent retaliation against opposing political faction members",
    "Using 2010 crackdown references to incite new violence against military",
    "Regional hate speech pitting North/NE against Bangkok voters",
    "Spreading fake news about political figures to provoke anger and mob action",
]


async def generate_for_label(
    label_config: dict,
    themes: list[str],
    llm_client: LLMClient,
    total_count: int,
    batch_size: int = 50,
    temperature: float = 0.9,
) -> list[dict]:
    """Generate samples for one Thailand label."""
    all_samples = []
    num_batches = (total_count + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        count = min(batch_size, total_count - len(all_samples))
        if count <= 0:
            break

        theme = themes[batch_idx % len(themes)]
        prompt = build_prompt(label_config, count, theme)

        temp = min(temperature + (batch_idx * 0.02), 1.0)
        samples = await generate_and_parse(llm_client, prompt, temp)

        for s in samples:
            s.setdefault("country_code", "TH")
            s.setdefault("label", label_config["label"])
            s.setdefault("is_violation", True)
            s.setdefault("severity", label_config["severity"])
            s.setdefault("detection_type", label_config["detection_type"])
            s.setdefault("language", "th")
            s.setdefault("generation_strategy", "positive")
            s.setdefault("adversarial_technique", None)

        all_samples.extend(samples)
        logger.info(f"  Batch {batch_idx+1}/{num_batches}: "
                    f"got {len(samples)} samples (total: {len(all_samples)})")

    return all_samples


async def generate_all(
    lese_majeste_count: int = 800,
    political_count: int = 400,
    config_path: str = "config.yaml",
):
    config = load_config(config_path)
    llm_client = LLMClient(config.llm, config.proxy)

    logger.info(f"Target: {lese_majeste_count} lese_majeste + {political_count} political_instigation")
    out_dir = OUT_DIR / "TH"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Generate lese_majeste samples
        logger.info(f"\n{'='*50}")
        logger.info("Generating local_th_lese_majeste samples...")
        logger.info(f"{'='*50}")
        lm_samples = await generate_for_label(
            LESE_MAJESTE, LESE_MAJESTE_THEMES, llm_client,
            total_count=lese_majeste_count, batch_size=50,
        )

        if lm_samples:
            fpath = out_dir / "local_th_lese_majeste.jsonl"
            with open(fpath, "a", encoding="utf-8") as f:
                for s in lm_samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            logger.info(f"  Wrote {len(lm_samples)} lese_majeste samples → {fpath}")

        # Generate political_instigation samples
        logger.info(f"\n{'='*50}")
        logger.info("Generating local_th_political_instigation samples...")
        logger.info(f"{'='*50}")
        pi_samples = await generate_for_label(
            POLITICAL_INSTIGATION, POLITICAL_THEMES, llm_client,
            total_count=political_count, batch_size=50,
        )

        if pi_samples:
            fpath = out_dir / "local_th_political_instigation.jsonl"
            with open(fpath, "a", encoding="utf-8") as f:
                for s in pi_samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            logger.info(f"  Wrote {len(pi_samples)} political_instigation samples → {fpath}")

        total = len(lm_samples) + len(pi_samples)
        logger.info(f"\nGRAND TOTAL: {total} TH samples generated")
    finally:
        await llm_client.close()

    return total


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Thailand local label samples via LLM")
    parser.add_argument("--samples-lese-majeste", type=int, default=800)
    parser.add_argument("--samples-political", type=int, default=400)
    parser.add_argument("--config", "-c", default="config.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    asyncio.run(generate_all(
        lese_majeste_count=args.samples_lese_majeste,
        political_count=args.samples_political,
        config_path=args.config,
    ))
