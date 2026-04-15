"""
Butterfly Classifier Backbone
==============================
Progressive architecture supporting ablation of each component:

  Phase 1: ConvNeXt-Tiny (ImageNet pretrained) + FC head         → Baseline
  Phase 2: + Coordinate Attention after each stage                → +CA
  Phase 3: + MLFI multi-level feature fusion (2:4:1:8)           → +MLFI
  Phase 4: (Focal Loss — applied in train.py, not here)          → +FL
  Phase 5: + Geotemporal late fusion (optional)                  → +Geo
  Phase 7: + Dual head (prototypical for sparse, conditional)    → +Proto

Each component is toggled by a flag, enabling systematic ablation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, List

from .coord_attention import CoordinateAttention
from .mlfi import MLFIModule
from .geotemporal import GeottemporalFusion


class ButterflyClassifier(nn.Module):
    """Full butterfly classification model with progressive components.

    Args:
        num_classes: Number of butterfly species (208)
        backbone_name: timm model name (default 'convnext_tiny')
        pretrained: Use ImageNet pretrained weights
        use_ca: Enable Coordinate Attention after each ConvNeXt stage
        use_mlfi: Enable Multi-Level Feature Interaction module
        use_geotemporal: Enable geotemporal feature fusion
        dropout: Dropout rate before classification head
        ca_reduction: Channel reduction ratio for CA modules
    """

    # ConvNeXt-Tiny stage output channels
    STAGE_CHANNELS = [96, 192, 384, 768]

    def __init__(
        self,
        num_classes: int = 208,
        backbone_name: str = 'convnext_tiny',
        pretrained: bool = True,
        use_ca: bool = False,
        use_mlfi: bool = False,
        use_geotemporal: bool = False,
        dropout: float = 0.3,
        ca_reduction: int = 32,
    ):
        super().__init__()
        self.use_ca = use_ca
        self.use_mlfi = use_mlfi
        self.use_geotemporal = use_geotemporal
        self.num_classes = num_classes

        # ─── Backbone ────────────────────────────────────────────────────
        # Load ConvNeXt-Tiny with all stages accessible
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,           # Return intermediate features
            out_indices=(0, 1, 2, 3),     # All 4 stages
        )

        # ─── Coordinate Attention (Phase 2) ──────────────────────────────
        if use_ca:
            self.ca_modules = nn.ModuleList([
                CoordinateAttention(ch, reduction=ca_reduction)
                for ch in self.STAGE_CHANNELS
            ])
        else:
            self.ca_modules = None

        # ─── Feature Dimension Routing ───────────────────────────────────
        if use_mlfi:
            # Phase 3: MLFI concatenates multi-level features → 1440-dim
            self.mlfi = MLFIModule(
                stage_channels=self.STAGE_CHANNELS,
                proportions=[2, 4, 1, 8],
                base_dim=96,
            )
            visual_dim = self.mlfi.total_dim  # 1440
        else:
            # Phase 1/2: Use only the final stage features
            self.mlfi = None
            self.pool = nn.AdaptiveAvgPool2d(1)
            visual_dim = self.STAGE_CHANNELS[-1]  # 768

        # ─── Geotemporal Fusion (Phase 5, optional) ──────────────────────
        if use_geotemporal:
            self.geo_fusion = GeottemporalFusion(
                visual_dim=visual_dim,
                zone_embed_dim=32,
                month_project_dim=0,  # Raw 2D cyclic
            )
            head_dim = self.geo_fusion.output_dim  # 1440 + 32 + 2 = 1474
        else:
            self.geo_fusion = None
            head_dim = visual_dim

        # ─── Classification Head ─────────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(head_dim),
            nn.Dropout(dropout),
            nn.Linear(head_dim, num_classes),
        )

        # Store feature dim for external access (e.g., prototypical head)
        self.feature_dim = head_dim

    def extract_features(
        self,
        x: torch.Tensor,
        zone_idx: Optional[torch.Tensor] = None,
        month_enc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract feature vector before classification head.

        Useful for:
          - Prototypical head (computing class prototypes)
          - Grad-CAM / embedding visualization
          - Transfer learning

        Returns:
            (B, feature_dim) feature vector
        """
        # Get multi-stage features from backbone
        stage_features = self.backbone(x)  # List of 4 tensors

        # Apply Coordinate Attention if enabled
        if self.ca_modules is not None:
            stage_features = [
                ca(feat) for ca, feat in zip(self.ca_modules, stage_features)
            ]

        # Feature extraction path
        if self.mlfi is not None:
            # MLFI: fuse all 4 stage features → (B, 1440)
            features = self.mlfi(stage_features)
        else:
            # Simple: pool final stage only → (B, 768)
            features = self.pool(stage_features[-1]).flatten(1)

        # Geotemporal fusion if enabled
        if self.geo_fusion is not None and zone_idx is not None and month_enc is not None:
            features = self.geo_fusion(features, zone_idx, month_enc)

        return features

    def forward(
        self,
        x: torch.Tensor,
        zone_idx: Optional[torch.Tensor] = None,
        month_enc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) input images
            zone_idx: (B,) biogeographic zone indices (optional)
            month_enc: (B, 2) cyclic month encoding (optional)
        Returns:
            (B, num_classes) logits
        """
        features = self.extract_features(x, zone_idx, month_enc)
        logits = self.head(features)
        return logits


def build_model(
    num_classes: int = 208,
    phase: int = 1,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> ButterflyClassifier:
    """Build model for a specific training phase.

    Args:
        num_classes: Number of butterfly species
        phase: Training phase (1=baseline, 2=+CA, 3=+MLFI, 5=+geo)
        pretrained: Use ImageNet pretrained backbone
        dropout: Dropout rate
    Returns:
        Configured ButterflyClassifier
    """
    config = {
        1: dict(use_ca=False, use_mlfi=False, use_geotemporal=False),
        2: dict(use_ca=True,  use_mlfi=False, use_geotemporal=False),
        3: dict(use_ca=True,  use_mlfi=True,  use_geotemporal=False),
        5: dict(use_ca=True,  use_mlfi=True,  use_geotemporal=True),
    }
    if phase not in config:
        raise ValueError(f"Unknown phase {phase}. Available: {list(config.keys())}")

    model = ButterflyClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **config[phase],
    )

    # Print model summary
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model Phase {phase}: {model.__class__.__name__}")
    print(f"  Components: CA={config[phase]['use_ca']}, "
          f"MLFI={config[phase]['use_mlfi']}, "
          f"Geo={config[phase]['use_geotemporal']}")
    print(f"  Parameters: {n_params:,} total, {n_trainable:,} trainable")
    print(f"  Feature dim: {model.feature_dim}")
    print(f"  Output classes: {num_classes}")

    return model
