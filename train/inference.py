"""
模型推理/测试模块 — 对输入文本进行违规检测

Usage:
  python -m train.inference --model trained_models/lora-TH --text "测试文本"
  python -m train.inference --model trained_models/lora-TH --file test_inputs.txt
  python -m train.inference --model trained_models/lora-TH --csv test_inputs.csv --output results.csv
"""

import argparse
import csv
import json
import logging
import os
import sys
import torch
from pathlib import Path

logger = logging.getLogger(__name__)


def load_model(save_dir: str, base_model_name: str, device: torch.device):
    """Load a trained country model and its metadata."""
    from .model import CountrySafetyClassifier

    info_path = os.path.join(save_dir, "model_info.json")
    results_path = os.path.join(save_dir, "results.json")

    # Load model info (label list, num_labels)
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        num_labels = info.get("num_labels", 8)
    else:
        # Fallback: load from results.json
        with open(results_path) as f:
            results = json.load(f)
        num_labels = results["num_labels"]

    label_list = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        label_list = results.get("label_list", [])

    model = CountrySafetyClassifier.from_pretrained(save_dir, base_model_name, num_labels)
    model.to(device)
    model.eval()
    return model, label_list


def predict(model, tokenizer, texts: list[str], label_list: list[str],
            device: torch.device, threshold: float = 0.5, max_length: int = 256
            ) -> list[dict]:
    """Run inference on a list of texts. Returns list of prediction dicts."""
    results = []
    model.eval()

    with torch.no_grad():
        for text in texts:
            encoding = tokenizer(
                text or "",
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

            result = {"text": text}
            triggered = []
            for i, label in enumerate(label_list):
                result[f"prob_{label}"] = float(probs[i])
                result[f"flag_{label}"] = bool(probs[i] >= threshold)
                if probs[i] >= threshold:
                    triggered.append(label)

            result["is_violation"] = len(triggered) > 0
            result["triggered_labels"] = "; ".join(triggered)
            results.append(result)

    return results


def predict_to_csv(model, tokenizer, texts: list[str], label_list: list[str],
                   device: torch.device, output_path: str, threshold: float = 0.5):
    """Run inference and save results to CSV."""
    results = predict(model, tokenizer, texts, label_list, device, threshold)

    # Build column names
    prob_cols = [f"prob_{l}" for l in label_list]
    flag_cols = [f"flag_{l}" for l in label_list]
    fieldnames = ["text", "is_violation", "triggered_labels"] + prob_cols + flag_cols

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    triggered_count = sum(1 for r in results if r["is_violation"])
    print(f"Results saved to {output_path}")
    print(f"  Total: {len(results)} texts")
    print(f"  Flagged: {triggered_count} ({100*triggered_count/max(1,len(results)):.1f}%)")
    print(f"  Labels: {label_list}")

    # Per-label stats
    for label in label_list:
        flagged = sum(1 for r in results if r.get(f"flag_{label}"))
        if flagged > 0:
            print(f"    {label}: {flagged} flagged")

    return results


def main():
    parser = argparse.ArgumentParser(description="模型推理/测试")
    parser.add_argument("--model", "-m", required=True, help="训练好的模型目录，如 trained_models/lora-TH")
    parser.add_argument("--base-model", default="microsoft/mdeberta-v3-base", help="基座模型路径")
    parser.add_argument("--text", help="单条文本推理")
    parser.add_argument("--file", help="文本文件，每行一条")
    parser.add_argument("--csv", help="CSV 输入文件（需含 text 列）")
    parser.add_argument("--output", "-o", default="predictions.csv", help="输出 CSV 路径")
    parser.add_argument("--threshold", type=float, default=0.5, help="判定阈值")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)

    # Load model
    print(f"Loading model from {args.model}...")
    model, label_list = load_model(args.model, args.base_model, device)
    print(f"Labels: {label_list}")

    # Load saved training results if available
    results_path = os.path.join(args.model, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            train_results = json.load(f)
        print(f"\nTrained model performance:")
        print(f"  Best val F1: {train_results['best_val_f1']:.4f}")
        if "test_metrics" in train_results:
            tm = train_results["test_metrics"]
            print(f"  Test Macro F1: {tm['macro_f1']:.4f}")
            print(f"  Test Macro Precision: {tm['macro_precision']:.4f}")
            print(f"  Test Macro Recall: {tm['macro_recall']:.4f}")
            for label in label_list:
                if f"{label}_f1" in tm:
                    print(f"    {label}: F1={tm[f'{label}_f1']:.4f}")

    # Collect input texts
    texts = []
    if args.text:
        texts = [args.text]
    elif args.file:
        with open(args.file) as f:
            texts = [line.strip() for line in f if line.strip()]
    elif args.csv:
        with open(args.csv, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            texts = [row.get("text", "") for row in reader if row.get("text", "").strip()]
    else:
        print("请输入测试文本 (Ctrl+D 结束):")
        texts = [line.strip() for line in sys.stdin if line.strip()]

    if not texts:
        print("No input text provided.")
        return

    # Run inference
    if len(texts) == 1 and args.text:
        # Single text: print detailed results
        results = predict(model, tokenizer, texts, label_list, device, args.threshold)
        r = results[0]
        print(f"\nText: {r['text'][:200]}...")
        print(f"Violation: {r['is_violation']}")
        if r['triggered_labels']:
            print(f"Triggered: {r['triggered_labels']}")
        print("\nPer-label probabilities:")
        for label in label_list:
            prob = r[f"prob_{label}"]
            flag = "⚠️" if r[f"flag_{label}"] else "✓"
            bar = "█" * int(prob * 20)
            print(f"  {flag} {label:<45} {prob:.4f} {bar}")
    else:
        predict_to_csv(model, tokenizer, texts, label_list, device, args.output, args.threshold)


if __name__ == "__main__":
    main()
