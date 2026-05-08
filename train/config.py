import yaml
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    base_model: str = "xlm-roberta-base"
    max_length: int = 256
    batch_size: int = 16
    learning_rate: float = 2e-4
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    eval_steps: int = 20
    logging_steps: int = 10
    save_best: bool = True
    early_stopping_patience: int = 3
    seed: int = 42
    cpu_batch_size: int = 8


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: list = field(default_factory=lambda: ["query", "value"])


@dataclass
class DataConfig:
    input_dir: str = "output"
    taxonomy_dir: str = "taxonomy"
    output_dir: str = "trained_models"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    min_samples_per_label: int = 50
    pos_neg_ratio: float = 1.0


@dataclass
class FocalLossConfig:
    gamma_map: dict = field(default_factory=lambda: {
        "keyword_sensitive": 0.0,
        "contextual": 2.0,
        "hybrid": 1.0,
    })
    alpha: float = 0.25
    reduction: str = "mean"


@dataclass
class TrainAppConfig:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    focal_loss: FocalLossConfig = field(default_factory=FocalLossConfig)


def _coerce_types(d: dict, float_keys: list[str], int_keys: list[str]) -> dict:
    """Force type conversion for values that YAML might parse as strings."""
    result = dict(d)
    for k in float_keys:
        if k in result and not isinstance(result[k], float):
            result[k] = float(result[k])
    for k in int_keys:
        if k in result and not isinstance(result[k], int):
            result[k] = int(result[k])
    return result


def load_train_config(config_path: str = "train_config.yaml") -> TrainAppConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    training_raw = _coerce_types(
        raw.get("training", {}),
        float_keys=["learning_rate", "warmup_ratio", "weight_decay"],
        int_keys=["max_length", "batch_size", "num_epochs", "eval_steps", "logging_steps",
                   "early_stopping_patience", "seed", "cpu_batch_size"],
    )
    lora_raw = _coerce_types(
        raw.get("lora", {}),
        float_keys=["dropout"],
        int_keys=["r", "alpha"],
    )
    data_raw = _coerce_types(
        raw.get("data", {}),
        float_keys=["train_ratio", "val_ratio", "test_ratio", "pos_neg_ratio"],
        int_keys=["min_samples_per_label"],
    )
    focal_raw = raw.get("focal_loss", {})
    focal_raw = _coerce_types(focal_raw, float_keys=["alpha"], int_keys=[])

    return TrainAppConfig(
        training=TrainingConfig(**training_raw),
        lora=LoRAConfig(**lora_raw),
        data=DataConfig(**data_raw),
        focal_loss=FocalLossConfig(**focal_raw),
    )
