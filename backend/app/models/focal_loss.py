"""Focal Loss for Imbalanced Crop Disease Segmentation and Detection.

Standard Cross-Entropy loss is dominated by easy negative pixels (healthy crops & soil)
which comprise >90% of agricultural terrain. Focal Loss reshapes the loss surface
by adding a modulating factor (1 - p_t)^gamma to downweight easy examples and
focus learning on hard, subtle pre-symptomatic fungal infection anomalies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union


class FocalLoss(nn.Module):
    """Multi-class Focal Loss supporting (B, C, H, W) segmentation and (B, C) classification."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[List[float], torch.Tensor]] = None,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
            else:
                self.register_buffer("alpha", alpha.clone().detach().to(dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Focal Loss.

        Args:
            logits: (B, C, H, W) or (B, C) unnormalized raw predictions.
            targets: (B, H, W) or (B,) ground truth class indices in [0, C-1].
        """
        # Calculate log-probabilities
        log_p = F.log_softmax(logits, dim=1)
        p = torch.exp(log_p)

        # Handle 4D segmentation maps: reshape (B, C, H, W) -> (B*H*W, C)
        if logits.dim() == 4:
            B, C, H, W = logits.shape
            logits_flat = logits.permute(0, 2, 3, 1).contiguous().view(-1, C)
            targets_flat = targets.view(-1)
            log_p_flat = log_p.permute(0, 2, 3, 1).contiguous().view(-1, C)
            p_flat = p.permute(0, 2, 3, 1).contiguous().view(-1, C)
        elif logits.dim() == 2:
            logits_flat = logits
            targets_flat = targets
            log_p_flat = log_p
            p_flat = p
        else:
            raise ValueError(f"Expected 2D or 4D logits tensor, got {logits.shape}")

        # Mask out ignore_index
        valid_mask = targets_flat != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        targets_valid = targets_flat[valid_mask]
        log_p_valid = log_p_flat[valid_mask]
        p_valid = p_flat[valid_mask]

        # Gather target probabilities
        target_log_p = log_p_valid.gather(dim=1, index=targets_valid.unsqueeze(1)).squeeze(1)
        target_p = p_valid.gather(dim=1, index=targets_valid.unsqueeze(1)).squeeze(1)

        # Focal modulating factor: (1 - p_t)^gamma
        modulating_factor = torch.pow(1.0 - target_p, self.gamma)

        focal_loss = -modulating_factor * target_log_p

        # Apply class weights alpha
        if self.alpha is not None:
            alpha_t = self.alpha[targets_valid]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

