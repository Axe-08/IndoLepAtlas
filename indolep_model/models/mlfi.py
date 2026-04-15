"""
Multi-Level Feature Interaction (MLFI) Module
=============================================
From: "FGBNet: Fine-Grained Bio-Subspecies Network" (Yuan et al., 2025)

Extracts features from all 4 backbone stages and concatenates them with
learned proportions, preventing the loss of discriminative low-level
texture details (wing banding, submarginal patterns, cell-spots) that
are progressively lost through deep pooling.

Optimal proportion: 2:4:1:8 (n1=192, n2=384, n3=96, n4=768 → total 1440)
This emphasizes semantic features (n4 ~53%) while preserving mid-level
wing patterns (n2 ~27%).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class DISBranch(nn.Module):
    """Detail Information Supplement (DIS) Branch.

    One branch per backbone stage. Applies adaptive max pooling
    followed by a fully connected layer for feature screening.

    Uses max pooling (not average) to retain the most salient
    activations — the strongest wing-pattern responses.
    """

    def __init__(self, in_channels: int, out_features: int, pool_size: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveMaxPool2d((pool_size, pool_size))
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(in_channels * pool_size * pool_size, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map from a backbone stage (B, C, H, W)
        Returns:
            Feature vector (B, out_features)
        """
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class MLFIModule(nn.Module):
    """Multi-Level Feature Interaction Module.

    Concatenates DIS branch outputs from all 4 backbone stages
    using the CONCAT strategy (not ADD or MULTI, which show
    overfitting per FGBNet ablation).

    Args:
        stage_channels: List of channel dims for each stage
            ConvNeXt-Tiny: [96, 192, 384, 768]
        proportions: Relative proportions for output dims (default 2:4:1:8)
        base_dim: Base dimension unit (proportions × base_dim = output dims)
        pool_size: Spatial size for adaptive pooling in each DIS branch
    """

    def __init__(
        self,
        stage_channels: List[int] = [96, 192, 384, 768],
        proportions: List[int] = [2, 4, 1, 8],
        base_dim: int = 96,
        pool_size: int = 4,
    ):
        super().__init__()
        self.out_dims = [p * base_dim for p in proportions]
        # n1=192, n2=384, n3=96, n4=768 → total = 1440

        self.branches = nn.ModuleList([
            DISBranch(ch, out_d, pool_size)
            for ch, out_d in zip(stage_channels, self.out_dims)
        ])
        self.total_dim = sum(self.out_dims)

    def forward(self, stage_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            stage_features: List of 4 feature maps, one per backbone stage
                [(B, 96, H1, W1), (B, 192, H2, W2), (B, 384, H3, W3), (B, 768, H4, W4)]
        Returns:
            Concatenated feature vector (B, 1440)
        """
        assert len(stage_features) == len(self.branches), \
            f"Expected {len(self.branches)} stage features, got {len(stage_features)}"

        branch_outputs = [
            branch(feat) for branch, feat in zip(self.branches, stage_features)
        ]
        return torch.cat(branch_outputs, dim=1)
