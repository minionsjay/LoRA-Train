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

        # Load base model — use fp16 on GPU to save VRAM, fp32 on CPU
        load_kwargs = {}
        if torch.cuda.is_available():
            load_kwargs["torch_dtype"] = torch.float16
        self.encoder = AutoModel.from_pretrained(base_model_name, **load_kwargs)

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

        # Classification head
        hidden_size = AutoConfig.from_pretrained(base_model_name).hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels),
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"CountrySafetyClassifier: {trainable:,} trainable / {total:,} total params")

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token embedding for classification
        pooled = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled)
        return logits

    def save_pretrained(self, save_dir: str):
        """Save LoRA adapter + classifier head."""
        import os
        os.makedirs(save_dir, exist_ok=True)

        # Save LoRA weights
        self.encoder.save_pretrained(save_dir)

        # Save classifier head
        classifier_path = os.path.join(save_dir, "classifier.pt")
        torch.save(self.classifier.state_dict(), classifier_path)
        logger.info(f"Saved model to {save_dir}")

    @classmethod
    def from_pretrained(cls, save_dir: str, base_model_name: str, num_labels: int):
        """Load a saved model."""
        import os
        from peft import PeftModel

        # Load base model
        base = AutoModel.from_pretrained(base_model_name)
        for param in base.parameters():
            param.requires_grad = False

        # Load LoRA
        encoder = PeftModel.from_pretrained(base, save_dir)

        # Load classifier
        model = cls.__new__(cls)
        super(CountrySafetyClassifier, model).__init__()
        model.num_labels = num_labels
        model.encoder = encoder

        hidden_size = AutoConfig.from_pretrained(base_model_name).hidden_size
        model.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_labels),
        )
        classifier_path = os.path.join(save_dir, "classifier.pt")
        model.classifier.load_state_dict(torch.load(classifier_path, map_location="cpu"))
        model.classifier.eval()

        return model
