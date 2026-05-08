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
        train_pos_counts: dict[str, int] | None = None,
    ):
        super().__init__()
        self.label_list = label_list
        self.alpha = focal_config.alpha
        self.reduction = focal_config.reduction

        # Build gamma tensor: auto-reduce for low-sample labels
        # Labels with < 100 training samples get gamma=0 (standard BCE)
        # They don't have enough diversity for Focal Loss to help
        gammas = []
        gamma_info = {}
        for label in label_list:
            dt = detection_type_map.get(label, "contextual")
            gamma = focal_config.gamma_map.get(dt, 2.0)
            if train_pos_counts and train_pos_counts.get(label, 0) < 100:
                gamma = 0.0  # Use standard BCE for low-sample labels
            gammas.append(gamma)
            gamma_info[label] = gamma
        self.register_buffer("gammas", torch.tensor(gammas, dtype=torch.float))
        logger.info(f"Per-label gammas: {gamma_info}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute per-label Focal Loss — optimized for sparse multi-label.

        Key insight: in multi-label with 7-8 labels per sample, most label
        positions are 0. A naive mean() over all positions is dominated by
        easy negatives, leading the model to always predict 0.

        Solution: compute mean separately for positive and negative positions,
        then combine. This ensures positive samples contribute equally to loss.
        """
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        probs = torch.sigmoid(logits)

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        bce = torch.clamp(bce, max=50.0)

        pt = torch.where(targets == 1, probs, 1.0 - probs)
        focal_weight = (1.0 - pt) ** self.gammas.unsqueeze(0)

        alpha_t = torch.where(
            targets == 1,
            torch.full_like(pt, self.alpha),
            torch.full_like(pt, 1.0 - self.alpha),
        )

        loss = alpha_t * focal_weight * bce
        loss = torch.clamp(loss, max=100.0)

        # Per-label mean: average positive and negative contributions separately
        # This prevents the loss from being dominated by easy negatives
        pos_mask = (targets == 1)
        neg_mask = (targets == 0)

        pos_loss = loss[pos_mask].mean() if pos_mask.any() else torch.tensor(0.0, device=loss.device)
        neg_loss = loss[neg_mask].mean() if neg_mask.any() else torch.tensor(0.0, device=loss.device)

        # Weight: positives contribute more (alpha), negatives less (1-alpha)
        return self.alpha * pos_loss + (1.0 - self.alpha) * neg_loss


def get_loss_fn(
    label_list: list[str],
    detection_type_map: dict[str, str],
    focal_config: FocalLossConfig,
    train_pos_counts: dict[str, int] | None = None,
) -> PerLabelFocalLoss:
    return PerLabelFocalLoss(label_list, detection_type_map, focal_config, train_pos_counts)
