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
        """Compute per-label Focal Loss with NaN-safe implementation.

        Uses log-sum-exp trick internally via BCEWithLogitsLoss, then applies
        focal weight in a numerically stable way (clamping to avoid 0*inf).
        """
        # Clamp logits to prevent extreme values
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        probs = torch.sigmoid(logits)

        # BCE loss per element — numerically stable via log-sum-exp
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # p_t = probability of correct class
        pt = torch.where(targets == 1, probs, 1.0 - probs)
        # Clamp pt away from 0 to avoid (1-pt) = 1.0 → focal_weight = 1.0 is fine,
        # but pt = 0 → (1-pt)^gamma = 1^gamma = 1. Also fine.
        # The real issue: pt near 0 → bce very large → focal_weight * bce overflows
        # Fix: clamp bce before applying focal weight
        bce = torch.clamp(bce, max=50.0)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - pt) ** self.gammas.unsqueeze(0)

        # Alpha balancing
        alpha_t = torch.where(
            targets == 1,
            torch.full_like(pt, self.alpha),
            torch.full_like(pt, 1.0 - self.alpha),
        )

        loss = alpha_t * focal_weight * bce
        # Final safety clamp
        loss = torch.clamp(loss, max=100.0)

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
