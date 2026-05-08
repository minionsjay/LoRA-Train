import os
import json
import logging
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm

from .config import TrainAppConfig, TrainingConfig, LoRAConfig, FocalLossConfig
from .model import CountrySafetyClassifier
from .losses import PerLabelFocalLoss
from .dataset import prepare_country_data, SafetyDataset

logger = logging.getLogger(__name__)


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, label_list: list[str], threshold: float = 0.5) -> dict:
    """Compute per-label and macro-averaged metrics."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    logits_np = logits.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    preds_np = preds.detach().cpu().numpy()

    metrics = {}

    # Per-label metrics
    for i, label in enumerate(label_list):
        if targets_np[:, i].sum() > 0:  # Only compute if label has positive samples
            metrics[f"{label}_f1"] = float(f1_score(targets_np[:, i], preds_np[:, i], zero_division=0))
            metrics[f"{label}_precision"] = float(precision_score(targets_np[:, i], preds_np[:, i], zero_division=0))
            metrics[f"{label}_recall"] = float(recall_score(targets_np[:, i], preds_np[:, i], zero_division=0))

    # Macro averages
    metrics["macro_f1"] = float(f1_score(targets_np, preds_np, average="macro", zero_division=0))
    metrics["macro_precision"] = float(precision_score(targets_np, preds_np, average="macro", zero_division=0))
    metrics["macro_recall"] = float(recall_score(targets_np, preds_np, average="macro", zero_division=0))
    metrics["accuracy"] = float(accuracy_score(targets_np, preds_np))

    return metrics


def train_country(
    country_code: str,
    config: TrainAppConfig,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> str:
    """Train a LoRA model for one country. Returns path to saved model."""

    tc = config.training
    lc = config.lora
    dc = config.data
    fc = config.focal_loss

    logger.info(f"{'='*60}")
    logger.info(f"Training model for {country_code}")
    logger.info(f"{'='*60}")

    # Prepare data
    train_ds, val_ds, test_ds, label_list, detection_type_map = prepare_country_data(
        dc, country_code, tokenizer
    )
    num_labels = len(label_list)
    logger.info(f"Labels ({num_labels}): {label_list}")

    batch_size = tc.batch_size if device.type == "cuda" else tc.cpu_batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Build model
    model = CountrySafetyClassifier(
        base_model_name=tc.base_model,
        lora_config=lc,
        num_labels=num_labels,
    )
    model.to(device)

    # Loss function
    loss_fn = PerLabelFocalLoss(label_list, detection_type_map, fc)
    loss_fn.to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tc.learning_rate,
        weight_decay=tc.weight_decay,
    )
    total_steps = len(train_loader) * tc.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * tc.warmup_ratio),
        num_training_steps=total_steps,
    )

    # Training loop
    grad_accum_steps = max(1, 4 // batch_size)  # maintain effective batch of 4
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_f1": [], "val_loss": []}

    for epoch in range(tc.num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_steps = 0

        pbar = tqdm(train_loader, desc=f"[{country_code}] Epoch {epoch+1}/{tc.num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels) / grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item() * grad_accum_steps
            train_steps += 1
            pbar.set_postfix({"loss": f"{loss.item() * grad_accum_steps:.4f}"})

        avg_train_loss = train_loss / max(train_steps, 1)
        history["train_loss"].append(avg_train_loss)

        # Validate
        model.eval()
        val_loss = 0.0
        all_logits, all_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids, attention_mask)
                loss = loss_fn(logits, labels)
                val_loss += loss.item()

                all_logits.append(logits)
                all_targets.append(labels)

        avg_val_loss = val_loss / max(len(val_loader), 1)
        history["val_loss"].append(avg_val_loss)

        val_logits = torch.cat(all_logits, dim=0)
        val_targets = torch.cat(all_targets, dim=0)
        val_metrics = compute_metrics(val_logits, val_targets, label_list)

        val_f1 = val_metrics["macro_f1"]
        history["val_f1"].append(val_f1)

        logger.info(
            f"[{country_code}] Epoch {epoch+1}: train_loss={avg_train_loss:.4f} "
            f"val_loss={avg_val_loss:.4f} val_macro_f1={val_f1:.4f}"
        )

        # Log per-label metrics for key local labels
        for label in label_list:
            if label.startswith(f"local_") and f"{label}_f1" in val_metrics:
                logger.info(f"  {label}: f1={val_metrics[f'{label}_f1']:.4f}")

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0

            # Save best model
            save_dir = os.path.join(dc.output_dir, f"lora-{country_code}")
            model.save_pretrained(save_dir)
            logger.info(f"  Saved best model to {save_dir} (val_f1={val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= tc.early_stopping_patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best model for test evaluation
    best_save_dir = os.path.join(dc.output_dir, f"lora-{country_code}")
    model = CountrySafetyClassifier.from_pretrained(best_save_dir, tc.base_model, num_labels)
    model.to(device)
    model.eval()

    all_logits, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, attention_mask)
            all_logits.append(logits)
            all_targets.append(labels)

    test_logits = torch.cat(all_logits, dim=0)
    test_targets = torch.cat(all_targets, dim=0)
    test_metrics = compute_metrics(test_logits, test_targets, label_list)

    logger.info(f"[{country_code}] Test Results:")
    logger.info(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
    logger.info(f"  Macro Precision: {test_metrics['macro_precision']:.4f}")
    logger.info(f"  Macro Recall: {test_metrics['macro_recall']:.4f}")
    for label in label_list:
        if f"{label}_f1" in test_metrics:
            logger.info(f"  {label}: f1={test_metrics[f'{label}_f1']:.4f}")

    # Save results
    results = {
        "country_code": country_code,
        "num_labels": num_labels,
        "label_list": label_list,
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "test_metrics": test_metrics,
        "history": history,
        "timestamp": datetime.now().isoformat(),
    }
    results_path = os.path.join(best_save_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return best_save_dir


def train_all_countries(
    config: TrainAppConfig,
    country_codes: list[str] | None = None,
) -> dict[str, str]:
    """Train LoRA models for multiple countries."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Download model (prefer HF mirror in China)
    import os as _os
    hf_endpoint = _os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    logger.info(f"HF endpoint: {hf_endpoint}")
    tokenizer = AutoTokenizer.from_pretrained(config.training.base_model)

    if country_codes is None:
        # Auto-detect from data directory
        data_dir = Path(config.data.input_dir)
        country_codes = sorted(
            d.name for d in data_dir.iterdir()
            if d.is_dir() and len(list(d.glob("*.jsonl"))) > 0
        )

    logger.info(f"Training for countries: {country_codes}")

    results = {}
    for cc in country_codes:
        try:
            save_dir = train_country(cc, config, tokenizer, device)
            results[cc] = save_dir
        except Exception as e:
            logger.error(f"Failed to train {cc}: {e}", exc_info=True)
            results[cc] = None

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Training Summary")
    logger.info(f"{'='*60}")
    for cc, save_dir in results.items():
        if save_dir:
            results_path = os.path.join(save_dir, "results.json")
            if os.path.exists(results_path):
                with open(results_path) as f:
                    r = json.load(f)
                logger.info(f"  {cc}: best_val_f1={r['best_val_f1']:.4f}  test_macro_f1={r['test_metrics']['macro_f1']:.4f}")
        else:
            logger.info(f"  {cc}: FAILED")

    return results
