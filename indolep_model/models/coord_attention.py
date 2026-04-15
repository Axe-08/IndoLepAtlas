"""
Coordinate Attention Module
===========================
From: "Coordinate Attention for Efficient Mobile Network Design" (Hou et al., CVPR 2021)

Processes height and width dimensions independently, generating attention weights
that encode positional information. Critical for butterfly classification where
the LOCATION of a wing pattern (e.g., eyespot on forewing vs hindwing) is
taxonomically meaningful.

Inserted after each of the 4 ConvNeXt stages (not inside each block) following
FGBNet's finding that this matches performance while using only 4 modules
instead of 27.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordinateAttention(nn.Module):
    """Coordinate Attention module.

    Decomposes channel attention into two 1D feature encoding processes
    that aggregate features along the two spatial directions respectively.
    This allows the model to attend to both 'where' and 'what' — critical
    for fine-grained features like butterfly wing banding patterns.

    Args:
        in_channels: Number of input channels
        reduction: Channel reduction ratio (default 32)
    """

    def __init__(self, in_channels: int, reduction: int = 32):
        super().__init__()
        mid_channels = max(8, in_channels // reduction)

        # Shared 1x1 conv for reduction
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),  # Hardswish/SiLU as in original paper
        )

        # Separate 1x1 convs for height and width attention
        self.fc_h = nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        self.fc_w = nn.Conv2d(mid_channels, in_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Attention-weighted tensor of same shape
        """
        B, C, H, W = x.shape

        # Aggregate along width → (B, C, H, 1)
        x_h = F.adaptive_avg_pool2d(x, (H, 1))
        # Aggregate along height → (B, C, 1, W)
        x_w = F.adaptive_avg_pool2d(x, (1, W)).permute(0, 1, 3, 2)  # → (B, C, W, 1)

        # Concatenate along spatial dimension → (B, C, H+W, 1)
        y = torch.cat([x_h, x_w], dim=2)

        # Shared reduction
        y = self.reduce(y)

        # Split back into h and w components
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # → (B, mid_C, 1, W)

        # Independent attention maps
        a_h = torch.sigmoid(self.fc_h(x_h))  # (B, C, H, 1)
        a_w = torch.sigmoid(self.fc_w(x_w))  # (B, C, 1, W)

        # Apply attention
        return x * a_h * a_w
