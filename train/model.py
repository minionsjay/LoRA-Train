import logging
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType

from .config import LoRAConfig, TrainingConfig

logger = logging.getLogger(__name__)


def build_model(
    training_config: TrainingConfig,
    lora_config: LoRAConfig,
    num_labels: int,
) -> nn.Module:
    """Build a LoRA-adapted XLM-RoBERTa model for multi-label classification.

    Returns a model with:
    - Frozen base (xlm-roberta-base)
    - LoRA adapters injected into query/value
    - Multi-label classification head
    """
    model_name = training_config.base_model

    logger.info(f"Loading base model: {model_name}")
    base_model = AutoModel.from_pretrained(model_name)

    # Freeze all base model parameters
    for param in base_model.parameters():
        param.requires_grad = False

    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_config.r,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        target_modules=lora_config.target_modules,
        bias="none",
    )

    model = get_peft_model(base_model, peft_config)
    logger.info(f"LoRA applied. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model


class CountrySafetyClassifier(nn.Module):
    """Full model: LoRA-adapted base + multi-label classification head."""

    def __init__(
        self,
        base_model_name: str,
        lora_config: LoRAConfig,
        num_labels: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_labels = num_labels

        # Load base model from local path — fp32 for training stability
        self.encoder = AutoModel.from_pretrained(base_model_name, local_files_only=True)

        for param in self.encoder.parameters():
            param.requires_grad = False

        # Inject LoRA
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_config.r,
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
            target_modules=lora_config.target_modules,
            bias="none",
        )
        self.encoder = get_peft_model(self.encoder, peft_config)

        # Classification head — match encoder dtype (fp16 on GPU, fp32 on CPU)
        hidden_size = AutoConfig.from_pretrained(base_model_name).hidden_size
        encoder_dtype = next(self.encoder.parameters()).dtype
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels),
        ).to(dtype=encoder_dtype)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"CountrySafetyClassifier: {trainable:,} trainable / {total:,} total params")

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token embedding for classification
        pooled = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled)
        return logits.float()  # 确保 loss 计算使用 fp32，避免 Half/Float 混合精度问题

    def save_pretrained(self, save_dir: str):
        """Save LoRA adapter + classifier head.

        Uses torch.save for maximum portability across PEFT versions.
        """
        import os, json
        os.makedirs(save_dir, exist_ok=True)

        # Save LoRA weights (extract only trainable params)
        lora_state = {}
        for name, param in self.encoder.named_parameters():
            if param.requires_grad:
                lora_state[name] = param.data.cpu()
        torch.save(lora_state, os.path.join(save_dir, "lora_weights.pt"))

        # Save LoRA config
        peft_config = self.encoder.peft_config.get("default", self.encoder.peft_config)
        if hasattr(peft_config, "to_dict"):
            peft_dict = peft_config.to_dict()
        else:
            peft_dict = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.1, "target_modules": ["query_proj", "value_proj"]}
        with open(os.path.join(save_dir, "lora_config.json"), "w") as f:
            json.dump(peft_dict, f, indent=2)

        # Save classifier head
        torch.save(self.classifier.state_dict(), os.path.join(save_dir, "classifier.pt"))

        # Save label count
        with open(os.path.join(save_dir, "model_info.json"), "w") as f:
            json.dump({"num_labels": self.num_labels}, f)

        logger.info(f"Saved model to {save_dir}")

    @classmethod
    def from_pretrained(cls, save_dir: str, base_model_name: str, num_labels: int):
        """Load a saved model from manual checkpoint."""
        import os, json

        lora_path = os.path.join(save_dir, "lora_weights.pt")
        config_path = os.path.join(save_dir, "lora_config.json")
        classifier_path = os.path.join(save_dir, "classifier.pt")

        if not os.path.exists(lora_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at {save_dir}.\n"
                f"Contents: {os.listdir(save_dir) if os.path.isdir(save_dir) else 'directory not found'}\n"
                f"Training may have failed before saving any checkpoint."
            )

        # Load base model from local path — fp32 for training stability
        base = AutoModel.from_pretrained(base_model_name, local_files_only=True)
        for param in base.parameters():
            param.requires_grad = False

        # Load LoRA config and inject
        with open(config_path) as f:
            peft_dict = json.load(f)
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=peft_dict.get("r", 16),
            lora_alpha=peft_dict.get("lora_alpha", 32),
            lora_dropout=peft_dict.get("lora_dropout", 0.1),
            target_modules=peft_dict.get("target_modules", ["query_proj", "value_proj"]),
        )
        encoder = get_peft_model(base, peft_config)

        # Load LoRA weights
        lora_state = torch.load(lora_path, map_location="cpu")
        encoder_state = encoder.state_dict()
        for name, param in lora_state.items():
            if name in encoder_state:
                encoder_state[name] = param
        encoder.load_state_dict(encoder_state)

        # Build model
        model = cls.__new__(cls)
        super(CountrySafetyClassifier, model).__init__()
        model.num_labels = num_labels
        model.encoder = encoder

        encoder_dtype = next(encoder.parameters()).dtype
        hidden_size = AutoConfig.from_pretrained(base_model_name, local_files_only=True).hidden_size
        model.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_labels),
        ).to(dtype=encoder_dtype)

        state_dict = torch.load(classifier_path, map_location="cpu")
        state_dict = {k: v.to(dtype=encoder_dtype) for k, v in state_dict.items()}
        model.classifier.load_state_dict(state_dict)
        model.classifier.eval()

        return model
