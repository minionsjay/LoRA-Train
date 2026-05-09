"""
数据管理工具 — 生成 base 违规样本 + 合并导出 CSV + 增量添加

Usage:
    python -m generate.data_mgmt generate-base --countries TH,ID  # 生成 base 违规样本
    python -m generate.data_mgmt export-csv                       # 导出所有数据到 data_csv/
    python -m generate.data_mgmt stats                            # 数据统计
    python -m generate.data_mgmt add --file new_data.csv          # 导入外部数据
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

from .config import load_config
from .llm_client import LLMClient, generate_and_parse
from .taxonomy_loader import load_all_countries, load_country_codes
from .models import CountryTaxonomy

logger = logging.getLogger(__name__)

BASE_VIOLATIONS = [
    {
        "label": "base_violence_dangerous_behavior",
        "severity": "HIGH",
        "detection_type": "contextual",
        "description": "暴力与危险行为 (Violence & Dangerous Behavior)。包括：煽动针对个人或群体的暴力行为、传播武器/爆炸物制作方法、极端血腥/酷刑/虐杀的文字或图像描述、组织或参与暴力活动（如线下斗殴、骚乱）、对暴力受害者的二次伤害和羞辱。不包括合法抗议/战争新闻/格斗体育。",
        "triggers": [
            "煽动针对特定个人或群体的暴力行动",
            "传播武器/爆炸物/毒药制作方法",
            "对暴力行为的极端详细图文描述",
            "组织或号召线下暴力活动",
            "对暴力受害者的羞辱和二次伤害",
        ],
        "boundary": "战争新闻纪实报导、合法抗议活动报道、武术/格斗体育赛事讨论、公共安全警示信息",
    },
    {
        "label": "base_hate_speech_harassment",
        "severity": "HIGH",
        "detection_type": "hybrid",
        "description": "仇恨言论与严重骚扰 (Hate Speech & Harassment)。包括：基于种族/民族/宗教/性别/性取向/残疾/年龄等受保护特征的攻击、系统性骚扰（如网络暴力、人肉搜索/Doxxing）、对弱势群体的非人化(dehumanization)叙事。注意：此标签覆盖的是'通用'仇恨言论，各国有更细粒度的本地化标签（如南非的 K-word 种族歧视、印尼的 SARA 违规）。",
        "triggers": [
            "基于受保护特征的贬损称呼和侮辱",
            "号召排斥/隔离/驱逐特定群体",
            "人肉搜索（公开他人隐私信息以引发骚扰）",
            "使用动物/疾病/污名比喻将特定群体非人化",
            "系统性针对个人的大规模网络攻击协调",
        ],
        "boundary": "对公共政策的理性批评（不涉及对特定群体的仇恨）、学术讨论种族/性别议题、受害者讲述被歧视经历",
    },
    {
        "label": "base_sexual_content_child_safety",
        "severity": "CRITICAL",
        "detection_type": "hybrid",
        "description": "色情与儿童保护 (Sexual Content & Child Safety)。包括：儿童性剥削材料(CSAM)、任何涉及未成年人的性暗示或性化描述、未经同意的私密影像传播(Revenge Porn)、露骨成人色情内容在非成人区的公开分享、性服务公开广告。注意：各国对'露骨'的定义差异很大，此标签仅覆盖最明确的硬色情和儿童保护红线。边缘内容交由各国 Local Delta 判断。",
        "triggers": [
            "涉及未成年人的性化描述或暗示",
            "使用代称（如'萝莉''正太'）讨论儿童相关内容",
            "未经同意的私密影像传播或威胁传播",
            "在非成人区公开分享露骨色情内容",
            "性服务的公开招揽和广告",
        ],
        "boundary": "医学论坛讨论性健康、艺术评论中的裸体分析、性教育内容、成人色情的合规讨论（在合法框架内）",
    },
    {
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
        ],
        "boundary": "心理健康支持论坛的求助帖、表达悲伤但无自杀意图的帖子、心理咨询师的专业建议、哀悼逝者",
    },
    {
        "label": "base_spam_deceptive_practices",
        "severity": "LOW",
        "detection_type": "keyword_sensitive",
        "description": "欺诈与垃圾信息 (Spam & Deceptive Practices)。包括：钓鱼链接和虚假登录页面、虚假中奖/退款/快递通知、庞氏骗局/传销/虚假投资推广、批量刷屏广告和机器人信息、虚假客服/冒充官方账号、虚假医疗广告和'神药'推广。",
        "triggers": [
            "钓鱼链接和虚假登录页面",
            "虚假中奖/退款/快递通知诈骗",
            "庞氏骗局/传销/虚假加密货币投资",
            "批量刷屏广告和机器人信息",
            "冒充官方账号/虚假客服",
            "虚假医疗广告和'神药'推广",
        ],
        "boundary": "合法的商品促销广告、真实投资理财教育、官方系统通知、正规医疗机构的健康科普",
    },
]


def build_base_violation_prompt(base_v: dict, languages: list[str], country_name: str, count: int) -> dict:
    """Build a prompt for generating base violation samples."""
    system = (
        "You are a data synthesis engine for a global content safety classifier. "
        "Generate realistic, varied examples of content that violates a SPECIFIC global policy. "
        "The content must be natural, believable, and reflect how real users express this violation type. "
        "Include diverse writing styles: social media posts, forum comments, chat messages, etc. "
        "Output ONLY valid JSON objects, one per line. No markdown fences."
    )

    user = (
        f"Generate {count} positive (violation) training samples for the following global content safety label.\n\n"
        f"=== LABEL ===\n"
        f"Label: {base_v['label']}\n"
        f"Severity: {base_v['severity']}\n"
        f"Detection Type: {base_v['detection_type']}\n"
        f"Description: {base_v['description']}\n\n"
        f"=== TRIGGER PATTERNS ===\n"
        + "\n".join(f"- {t}" for t in base_v['triggers']) + "\n\n"
        f"=== GENERATION RULES ===\n"
        f"1. Languages: mix of {', '.join(languages)} with natural code-switching\n"
        f"2. {country_name} context: some samples should reference {country_name} locations, culture, or current events\n"
        f"3. Vary length: 10-200 words\n"
        f"4. Vary format: social media post, forum reply, chat message, comment, blog\n"
        f"5. For keyword_sensitive labels: include character substitutions and evasion techniques in ~40% of samples\n"
        f"6. For contextual labels: express the violation through implication and context, not just explicit keywords\n\n"
        f"=== BOUNDARY (DO NOT generate this) ===\n"
        f"{base_v['boundary']}\n\n"
        f"Output exactly {count} JSON objects, one per line:\n"
        f'{{"text": string, "label": "{base_v["label"]}", "is_violation": true, '
        f'"severity": "{base_v["severity"]}", "detection_type": "{base_v["detection_type"]}", '
        f'"language": string, "generation_strategy": "base_positive", '
        f'"metadata": {{"triggers_used": [string, ...]}}}}'
    )

    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }


async def generate_base_violations(
    country: CountryTaxonomy,
    llm_client: LLMClient,
    samples_per_label: int = 30,
    temperature: float = 0.9,
) -> list[dict]:
    """Generate base violation samples for one country."""
    all_samples = []
    languages = country.languages
    country_name = country.country_name

    for base_v in BASE_VIOLATIONS:
        logger.info(f"  Generating {samples_per_label} samples for {base_v['label']}...")
        prompt = build_base_violation_prompt(base_v, languages, country_name, samples_per_label)
        samples = await generate_and_parse(llm_client, prompt, temperature)

        for s in samples:
            s.setdefault("country_code", country.country_code)
            s.setdefault("label", base_v["label"])
            s.setdefault("is_violation", True)
            s.setdefault("severity", base_v["severity"])
            s.setdefault("detection_type", base_v["detection_type"])
            s.setdefault("language", languages[0])
            s.setdefault("generation_strategy", "base_positive")
            s.setdefault("adversarial_technique", None)

        all_samples.extend(samples)
        logger.info(f"    Got {len(samples)} samples for {base_v['label']}")

    return all_samples


def build_base_negative_prompt(base_v: dict, languages: list[str], country_name: str, count: int, neg_type: str) -> dict:
    """Build prompt for base label negative samples (boundary or general)."""
    if neg_type == "boundary":
        system = (
            "You are a boundary-case specialist for a global content safety classifier. "
            "Generate text that SUPERFICIALLY RELATES to a violation category but is NOT actually a violation. "
            "These hard negatives must be plausible and genuinely challenging for a classifier to distinguish. "
            "Output ONLY valid JSON objects, one per line. No markdown fences."
        )
        user = (
            f"Generate {count} negative (NON-violation) training samples that look similar to the violation "
            f"but are actually safe.\n\n"
            f"=== VIOLATION LABEL (DO NOT generate this type of content) ===\n"
            f"Label: {base_v['label']}\n"
            f"Description: {base_v['description']}\n\n"
            f"=== SAFE BOUNDARY (generate content IN this category) ===\n"
            f"{base_v['boundary']}\n\n"
            f"=== GENERATION RULES ===\n"
            f"1. Languages: mix of {', '.join(languages)} with natural code-switching\n"
            f"2. {country_name} context: reference {country_name} locations, culture, or current events\n"
            f"3. Vary perspectives: academic analysis, news reporting, personal narrative, cultural discussion, legal/medical professional\n"
            f"4. Vary length: 10-200 words\n"
            f"5. Every sample must be clearly SAFE per the boundary definition above\n"
            f"6. Some samples should use words/terms that overlap with the violation domain but in a clearly benign way\n\n"
            f"Output exactly {count} JSON objects, one per line:\n"
            f'{{"text": string, "label": "{base_v["label"]}", "is_violation": false, '
            f'"severity": "{base_v["severity"]}", "detection_type": "{base_v["detection_type"]}", '
            f'"language": string, "generation_strategy": "base_boundary_negative", "metadata": {{"rationale": string}}}}'
        )
    else:  # general non-violation
        system = (
            "You are a data generator for content safety classifier training. "
            "Generate completely benign, harmless social media posts that have NO relation to any content violation. "
            "The posts should be natural, realistic user-generated content in the specified languages. "
            "Output ONLY valid JSON objects, one per line. No markdown fences."
        )
        user = (
            f"Generate {count} completely benign and harmless social media posts, comments, or messages "
            f"in {', '.join(languages)} (mix naturally, include some code-switching where realistic).\n\n"
            f"Country context: {country_name}. Reference local places, culture, food, events where natural.\n"
            f"Topics: everyday life, food, technology, travel, family, sports, music, hobbies, work, school.\n\n"
            f"Output exactly {count} JSON objects, one per line:\n"
            f'{{"text": string, "label": "{base_v["label"]}", "is_violation": false, '
            f'"severity": "{base_v["severity"]}", "detection_type": "{base_v["detection_type"]}", '
            f'"language": string, "generation_strategy": "base_general_negative", '
            f'"metadata": {{"rationale": "Random benign content unrelated to any violation"}}}}'
        )

    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }


async def generate_base_negatives(
    country: CountryTaxonomy,
    llm_client: LLMClient,
    boundary_per_label: int = 15,
    general_per_label: int = 40,
    temperature: float = 0.75,
) -> list[dict]:
    """Generate negative samples for base labels (boundary + general non-violation)."""
    all_samples = []
    languages = country.languages
    country_name = country.country_name

    for base_v in BASE_VIOLATIONS:
        # Boundary negatives
        logger.info(f"  Generating {boundary_per_label} boundary negatives for {base_v['label']}...")
        bnd_prompt = build_base_negative_prompt(base_v, languages, country_name, boundary_per_label, "boundary")
        bnd_samples = await generate_and_parse(llm_client, bnd_prompt, temperature * 0.85)
        for s in bnd_samples:
            s.setdefault("country_code", country.country_code)
            s.setdefault("label", base_v["label"])
            s.setdefault("is_violation", False)
            s.setdefault("severity", base_v["severity"])
            s.setdefault("detection_type", base_v["detection_type"])
            s.setdefault("language", languages[0])
            s.setdefault("generation_strategy", "base_boundary_negative")
        all_samples.extend(bnd_samples)
        logger.info(f"    Got {len(bnd_samples)} boundary negatives")

        # General non-violation
        logger.info(f"  Generating {general_per_label} general negatives for {base_v['label']}...")
        gen_prompt = build_base_negative_prompt(base_v, languages, country_name, general_per_label, "general")
        gen_samples = await generate_and_parse(llm_client, gen_prompt, temperature * 0.7)
        for s in gen_samples:
            s.setdefault("country_code", country.country_code)
            s.setdefault("label", base_v["label"])
            s.setdefault("is_violation", False)
            s.setdefault("severity", base_v["severity"])
            s.setdefault("detection_type", base_v["detection_type"])
            s.setdefault("language", languages[0])
            s.setdefault("generation_strategy", "base_general_negative")
        all_samples.extend(gen_samples)
        logger.info(f"    Got {len(gen_samples)} general negatives")

    return all_samples


def export_to_csv(
    jsonl_dir: str = "output",
    output_dir: str = "data_csv",
):
    """Export all JSONL data to per-country CSV files, including stats."""
    jsonl_path = Path(jsonl_dir)
    csv_path = Path(output_dir)
    csv_path.mkdir(parents=True, exist_ok=True)

    all_rows = []
    per_country = {}

    # Load all JSONL
    for fpath in sorted(jsonl_path.rglob("*.jsonl")):
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue

                row = {
                    "country_code": d.get("country_code", ""),
                    "label": d.get("label", ""),
                    "is_violation": str(d.get("is_violation", "")).lower(),
                    "text": d.get("text", ""),
                    "severity": d.get("severity", ""),
                    "detection_type": d.get("detection_type", ""),
                    "language": d.get("language", ""),
                    "generation_strategy": d.get("generation_strategy", ""),
                    "adversarial_technique": d.get("adversarial_technique", "") or "",
                }
                all_rows.append(row)
                cc = row["country_code"]
                if cc not in per_country:
                    per_country[cc] = []
                per_country[cc].append(row)

    fieldnames = list(all_rows[0].keys()) if all_rows else []

    # Write per-country CSV
    for cc, rows in sorted(per_country.items()):
        fpath = csv_path / f"safety_data_{cc}.csv"
        with open(fpath, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        labels = sorted(set(r["label"] for r in rows))
        pos = sum(1 for r in rows if r["is_violation"] == "true" and r["generation_strategy"] in ("positive", "base_positive"))
        neg = sum(1 for r in rows if r["is_violation"] == "false")
        print(f"  {cc}.csv: {len(rows)} rows, {len(labels)} labels, {pos}+ {neg}-")

    # Write merged CSV
    merged_path = csv_path / "safety_training_data.csv"
    with open(merged_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nMerged: {merged_path} ({len(all_rows)} rows)")

    return all_rows, per_country


def show_stats(jsonl_dir: str = "output"):
    """Show per-country and per-label statistics."""
    jsonl_path = Path(jsonl_dir)

    total = 0
    for country_dir in sorted(jsonl_path.iterdir()):
        if not country_dir.is_dir():
            continue
        cc = country_dir.name
        print(f"\n{'='*50}")
        print(f"  {cc}")
        print(f"{'='*50}")

        cc_total = 0
        for fpath in sorted(country_dir.glob("*.jsonl")):
            label = fpath.stem
            pos = adv = neg = 0
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue
                    if d.get("is_violation"):
                        if d.get("generation_strategy") == "adversarial_augmentation":
                            adv += 1
                        else:
                            pos += 1
                    else:
                        neg += 1
            cc_total += pos + adv + neg
            total += pos + adv + neg
            print(f"    {label:<38} total={pos+adv+neg:<5} pos={pos:<4} adv={adv:<4} neg={neg:<4}")

        print(f"    {'─'*60}")
        print(f"    {'TOTAL':<38} {cc_total}")

    print(f"\n  GRAND TOTAL: {total}")


async def generate_base_main(args):
    """CLI handler: generate base violations + negatives."""
    config = load_config(args.config)
    llm_client = LLMClient(config.llm, config.proxy)

    if args.countries:
        codes = [c.strip() for c in args.countries.split(",")]
        countries = load_country_codes(codes)
    else:
        countries = load_all_countries()

    try:
        for country in countries:
            cc = country.country_code
            logger.info(f"Generating base data for {cc} ({country.country_name})...")
            out_dir = Path(config.output.dir) / cc
            out_dir.mkdir(parents=True, exist_ok=True)

            all_samples = []

            # Phase 1: Positive samples
            if not args.negatives_only:
                pos_samples = await generate_base_violations(
                    country, llm_client,
                    samples_per_label=args.samples_per_label,
                    temperature=config.generation.temperature,
                )
                logger.info(f"  Phase 1: {len(pos_samples)} positive samples")
                all_samples.extend(pos_samples)
            else:
                logger.info("  Phase 1: SKIPPED (negatives-only mode) — loading existing data")

            # Phase 2: Negative samples (boundary + general)
            boundary_count = max(10, args.samples_per_label // 3)
            general_count = max(30, args.samples_per_label)
            neg_samples = await generate_base_negatives(
                country, llm_client,
                boundary_per_label=boundary_count,
                general_per_label=general_count,
                temperature=config.generation.temperature,
            )
            logger.info(f"  Phase 2: {len(neg_samples)} negative samples")
            all_samples.extend(neg_samples)
            logger.info(f"  Total: {len(all_samples)} base samples for {cc}")

            # Group by label and write
            by_label = {}
            for s in all_samples:
                lbl = s["label"]
                by_label.setdefault(lbl, []).append(s)

            for lbl, rows in by_label.items():
                fpath = out_dir / f"{lbl}.jsonl"
                mode = "a" if not args.force else "w"
                with open(fpath, mode, encoding="utf-8") as f:
                    for row in rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                pos = sum(1 for r in rows if r.get("is_violation"))
                neg = sum(1 for r in rows if not r.get("is_violation"))
                logger.info(f"    {lbl}: {len(rows)} rows ({pos}+ {neg}-) → {fpath}")

    finally:
        await llm_client.close()


def main():
    parser = argparse.ArgumentParser(description="数据管理工具")
    sub = parser.add_subparsers(dest="command")

    # generate-base
    p_gen = sub.add_parser("generate-base", help="生成 base 违规样本")
    p_gen.add_argument("--config", "-c", default="config.yaml")
    p_gen.add_argument("--countries", help="逗号分隔的国家代码")
    p_gen.add_argument("--samples-per-label", type=int, default=30)
    p_gen.add_argument("--force", action="store_true")
    p_gen.add_argument("--negatives-only", action="store_true", help="仅生成负样本（跳过正样本）")

    # export-csv
    p_exp = sub.add_parser("export-csv", help="导出 JSONL 到 CSV")
    p_exp.add_argument("--input-dir", default="output")
    p_exp.add_argument("--output-dir", default="data_csv")

    # stats
    sub.add_parser("stats", help="显示数据统计")

    # add (import external CSV)
    p_add = sub.add_parser("add", help="导入外部 CSV 数据")
    p_add.add_argument("--file", required=True, help="要导入的 CSV 文件")
    p_add.add_argument("--output-dir", default="output")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    if args.command == "generate-base":
        asyncio.run(generate_base_main(args))
    elif args.command == "export-csv":
        export_to_csv(args.input_dir, args.output_dir)
    elif args.command == "stats":
        show_stats(args.input_dir if hasattr(args, 'input_dir') else "output")
    elif args.command == "add":
        _import_csv(args.file, args.output_dir)
    else:
        parser.print_help()


def _import_csv(csv_file: str, output_dir: str = "output"):
    """Import external CSV data and convert to JSONL."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(csv_file, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    by_country_label = {}
    for row in rows:
        cc = row.get("country_code", "UN")
        label = row.get("label", "unknown")
        key = (cc, label)
        by_country_label.setdefault(key, []).append(row)

    for (cc, label), items in by_country_label.items():
        country_dir = out_path / cc
        country_dir.mkdir(parents=True, exist_ok=True)
        fpath = country_dir / f"{label}.jsonl"
        mode = "a"
        with open(fpath, mode, encoding="utf-8") as f:
            for item in items:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")
        print(f"  Imported {len(items)} rows → {fpath}")

    print(f"Done. {len(rows)} rows imported.")


if __name__ == "__main__":
    main()
