"""
Loss Functions for Butterfly Classification
=============================================
Provides:
  - Standard cross-entropy (with optional label smoothing)
  - Focal loss for class-imbalanced training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) for handling class imbalance.
    
    Applies a modulating factor (1 - p_t)^gamma to down-weight
    easy (well-classified) examples and focus on hard ones.
    
    Args:
        weight: Per-class weights (inverse frequency), shape (C,)
        gamma: Focusing parameter. Higher = more focus on hard examples.
        label_smoothing: Label smoothing factor.
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer('weight', weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C) raw model outputs
            targets: (B,) class indices
        Returns:
            Scalar loss
        """
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )
        
        pt = torch.exp(-ce_loss)  # p_t = probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        return focal_loss.mean()


def build_loss(
    loss_type: str = 'ce',
    class_weights: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    label_smoothing: float = 0.1,
) -> nn.Module:
    """Factory function to build the training loss.
    
    Args:
        loss_type: 'ce' for cross-entropy, 'focal' for focal loss
        class_weights: Optional per-class weights for focal loss
        gamma: Focal loss gamma parameter
        label_smoothing: Label smoothing factor
    Returns:
        Loss module
    """
    if loss_type == 'focal':
        return FocalLoss(
            weight=class_weights,
            gamma=gamma,
            label_smoothing=label_smoothing,
        )
    else:
        return nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
        )
