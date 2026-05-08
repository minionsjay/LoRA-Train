"""
文化感知跨语言安全对齐 — 训练数据生成器

Usage:
    python -m generate --config config.yaml
    python -m generate -c config.yaml --countries TH,ID,SG
    python -m generate -c config.yaml --labels local_th_lese_majeste
    python -m generate -c config.yaml --dry-run
    python -m generate -c config.yaml --force
"""

import argparse
import asyncio
import logging
import random
import sys
from pathlib import Path

from .config import load_config
from .taxonomy_loader import load_all_countries, load_country_codes, filter_labels, count_all_labels
from .llm_client import LLMClient
from .output_writer import OutputWriter
from .generator import Generator


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")


def parse_args():
    p = argparse.ArgumentParser(
        description="文化感知跨语言安全对齐 — 训练数据生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m generate -c config.yaml                        # Generate all
  python -m generate -c config.yaml --countries TH,ID      # Specific countries
  python -m generate -c config.yaml --labels local_th_lese_majeste  # Specific label
  python -m generate -c config.yaml --dry-run              # Preview only
  python -m generate -c config.yaml --force                # Overwrite existing
        """,
    )
    p.add_argument("-c", "--config", default="config.yaml", help="Path to YAML config file")
    p.add_argument("-f", "--force", action="store_true", help="Overwrite existing output files")
    p.add_argument("--countries", help="Comma-separated country codes (default: all)")
    p.add_argument("--labels", help="Comma-separated label names (default: all)")
    p.add_argument("--dry-run", action="store_true", help="Show generation plan without making API calls")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()


async def main_async(args):
    setup_logging(args.verbose)
    logger = logging.getLogger("generate")

    # Load config
    config_path = args.config
    logger.info(f"Loading config from {config_path}")
    config = load_config(config_path)

    # CLI flags override config
    if args.force:
        config.generation.force = True

    if config.generation.seed is not None:
        random.seed(config.generation.seed)

    # Load taxonomy
    if args.countries:
        codes = [c.strip() for c in args.countries.split(",")]
        countries = load_country_codes(codes)
    else:
        countries = load_all_countries()

    if args.labels:
        labels = [l.strip() for l in args.labels.split(",")]
        countries = filter_labels(countries, labels)

    logger.info(f"Loaded {len(countries)} countries with {sum(len(c.local_violations) for c in countries)} labels")

    # Dry run
    if args.dry_run:
        label_counts = count_all_labels(countries)
        total = sum(label_counts.values())
        logger.info("=== DRY RUN ===")
        for label, count in sorted(label_counts.items()):
            logger.info(f"  {label}: ~{count} samples")
        logger.info(f"Total: ~{total} samples across {len(label_counts)} labels")
        est_tokens = total * 80  # rough estimate: 80 tokens per sample
        logger.info(f"Estimated API calls: ~{len(label_counts) * 3} (positive + negative + adversarial)")
        logger.info(f"Estimated tokens: ~{est_tokens:,}")
        return

    # Initialize clients
    llm_client = LLMClient(config.llm, config.proxy)
    output_writer = OutputWriter(config.output.dir, force=config.generation.force)

    # Generate
    generator = Generator(config, llm_client, output_writer)
    try:
        await generator.generate_all(countries)
    finally:
        await llm_client.close()
        output_writer.close()
        logger.info("Done. Output files:")
        out_dir = Path(config.output.dir)
        for fpath in sorted(out_dir.rglob("*.jsonl")):
            logger.info(f"  {fpath}")


def main():
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
