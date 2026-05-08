"""
文化感知跨语言安全对齐 — 模型训练入口

Usage:
    python -m train                                # Train all countries
    python -m train --countries TH,ID              # Train specific countries
    python -m train --countries TH --epochs 5       # Custom epoch count
    python -m train --prepare-only                  # Show data stats only
    python -m train --eval-only --checkpoint trained_models/lora-TH  # Evaluate
"""

import argparse
import logging
import sys
import random
import numpy as np
import torch
from pathlib import Path

from .config import load_train_config
from .trainer import train_all_countries, train_country
from .dataset import prepare_country_data
from transformers import AutoTokenizer


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(
        description="文化感知跨语言安全对齐 — LoRA 训练器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m train                                    # Train all 8 countries
  python -m train --countries TH,ID,BR               # Train specific countries
  python -m train --countries TH --epochs 5          # Fewer epochs for testing
  python -m train --prepare-only                     # Show data statistics
  python -m train --eval-only --checkpoint trained_models/lora-TH  # Evaluate only
        """,
    )
    p.add_argument("--config", "-c", default="train_config.yaml", help="Training config YAML")
    p.add_argument("--countries", help="Comma-separated country codes (default: all)")
    p.add_argument("--epochs", type=int, help="Override num_epochs from config")
    p.add_argument("--prepare-only", action="store_true", help="Only show data statistics, no training")
    p.add_argument("--eval-only", action="store_true", help="Only evaluate a saved model")
    p.add_argument("--checkpoint", help="Model checkpoint path for --eval-only")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("train")

    config = load_train_config(args.config)
    set_seed(config.training.seed)

    if args.epochs:
        config.training.num_epochs = args.epochs

    # --prepare-only: Show data stats
    if args.prepare_only:
        tokenizer = AutoTokenizer.from_pretrained(config.training.base_model)
        if args.countries:
            codes = [c.strip() for c in args.countries.split(",")]
        else:
            data_dir = Path(config.data.input_dir)
            codes = sorted(d.name for d in data_dir.iterdir() if d.is_dir())

        for cc in codes:
            logger.info(f"\n{'='*50}")
            logger.info(f"Country: {cc}")
            train_ds, val_ds, test_ds, label_list, detection_types = prepare_country_data(
                config.data, cc, tokenizer
            )
            logger.info(f"  Labels: {label_list}")
            logger.info(f"  Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
            for label in label_list:
                train_pos = sum(
                    1 for i in range(len(train_ds))
                    if train_ds.labels[i][train_ds.label_to_idx[label]] > 0
                )
                logger.info(f"    {label}: train_pos={train_pos}")
        return

    # --eval-only: Evaluate saved model
    if args.eval_only:
        if not args.checkpoint:
            logger.error("--eval-only requires --checkpoint")
            sys.exit(1)
        # Load results if available
        import json, os
        results_path = os.path.join(args.checkpoint, "results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                r = json.load(f)
            logger.info(f"Saved results for {r['country_code']}:")
            logger.info(f"  Best val F1: {r['best_val_f1']:.4f} (epoch {r['best_epoch']})")
            logger.info(f"  Test macro F1: {r['test_metrics']['macro_f1']:.4f}")
            for label, f1_val in r['test_metrics'].items():
                if label.endswith('_f1'):
                    logger.info(f"  {label}: {f1_val:.4f}")
        else:
            logger.warning(f"No results.json found in {args.checkpoint}")
        return

    # --countries: Filter countries
    country_codes = None
    if args.countries:
        country_codes = [c.strip() for c in args.countries.split(",")]

    # Train
    results = train_all_countries(config, country_codes)

    # Final summary
    succeeded = sum(1 for v in results.values() if v is not None)
    failed = len(results) - succeeded
    logger.info(f"\nDone. {succeeded} succeeded, {failed} failed.")

    if succeeded > 0:
        logger.info(f"Models saved to: {config.data.output_dir}/")


if __name__ == "__main__":
    main()
