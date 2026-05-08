import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from .config import FocalLossConfig

logger = logging.getLogger(__name__)


class PerLabelFocalLoss(nn.Module):
    """Focal Loss with per-label gamma based on detection_type.

    Gamma values:
    - keyword_sensitive: 0.0 (standard BCE, easy patterns)
    - contextual: 2.0 (strong focal weight, hard semantic cases)
    - hybrid: 1.0 (moderate focal weight)
    """

    def __init__(
        self,
        label_list: list[str],
        detection_type_map: dict[str, str],
        focal_config: FocalLossConfig,
    ):
        super().__init__()
        self.label_list = label_list
        self.alpha = focal_config.alpha
        self.reduction = focal_config.reduction

        # Build gamma tensor: one value per label
        gammas = []
        for label in label_list:
            dt = detection_type_map.get(label, "contextual")
            gamma = focal_config.gamma_map.get(dt, 2.0)
            gammas.append(gamma)
        self.register_buffer("gammas", torch.tensor(gammas, dtype=torch.float))
        logger.info(f"Per-label gammas: {dict(zip(label_list, gammas))}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute per-label Focal Loss.

        Args:
            logits: (batch_size, num_labels) — raw logits
            targets: (batch_size, num_labels) — multi-hot binary targets

        Returns:
            scalar loss
        """
        probs = torch.sigmoid(logits)

        # BCE loss per element
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Focal weight: (1 - p_t)^gamma
        # p_t = probs if target=1, else 1-probs
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gammas.unsqueeze(0)

        # Alpha balancing for positive samples
        alpha_weight = torch.where(
            targets == 1,
            torch.tensor(self.alpha, device=targets.device),
            torch.tensor(1 - self.alpha, device=targets.device),
        )

        loss = alpha_weight * focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def get_loss_fn(
    label_list: list[str],
    detection_type_map: dict[str, str],
    focal_config: FocalLossConfig,
) -> PerLabelFocalLoss:
    return PerLabelFocalLoss(label_list, detection_type_map, focal_config)
