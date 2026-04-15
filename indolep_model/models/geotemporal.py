"""
Geotemporal Fusion Module (Optional Enhancement)
=================================================
Encodes geographic (biogeographic zone) and temporal (month) metadata
as auxiliary features for late fusion with visual features.

This is the paper's novel contribution — no prior butterfly/insect
classification paper incorporates geotemporal context.

Treated as OPTIONAL: the vision-only model must stand on its own.
Geotemporal features enhance accuracy at the decision boundary for
visually similar species with non-overlapping ranges/seasons.
"""

import torch
import torch.nn as nn
import numpy as np


class BiogeographicZoneEncoder(nn.Module):
    """Learned embedding for Indian biogeographic zones.

    Maps 9 zones (8 named + 1 unknown/missing) to a dense vector.
    The 'unknown' embedding acts like BERT's [MASK] token —
    a learned representation for missing location data.

    Args:
        num_zones: Number of biogeographic zones (default 9)
        embed_dim: Embedding dimension (default 32)
    """

    def __init__(self, num_zones: int = 9, embed_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(num_zones, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, zone_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            zone_idx: (B,) tensor of zone indices [0, num_zones)
        Returns:
            (B, embed_dim) zone embedding
        """
        return self.embedding(zone_idx)


class CyclicMonthEncoder(nn.Module):
    """Cyclic encoding of month with optional learned projection.

    Base encoding: [sin(2πM/12), cos(2πM/12)] — correctly places
    December adjacent to January.

    Optionally projects through a linear layer to a higher dimension.

    Args:
        project_dim: If > 0, project cyclic encoding to this dim.
                     If 0, output raw 2D cyclic encoding.
    """

    def __init__(self, project_dim: int = 0):
        super().__init__()
        self.input_dim = 2
        if project_dim > 0:
            self.project = nn.Sequential(
                nn.Linear(2, project_dim),
                nn.ReLU(inplace=True),
            )
            self.output_dim = project_dim
        else:
            self.project = None
            self.output_dim = 2

    def forward(self, month_enc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            month_enc: (B, 2) tensor of [sin, cos] month encodings
                       (precomputed in dataset.py)
        Returns:
            (B, output_dim) month encoding
        """
        if self.project is not None:
            return self.project(month_enc)
        return month_enc


class GeottemporalFusion(nn.Module):
    """Late fusion of geotemporal features with visual features.

    Concatenates zone embedding + month encoding with the visual
    feature vector before the classification head.

    Args:
        visual_dim: Dimension of visual feature vector (e.g., 1440 from MLFI)
        zone_embed_dim: Zone embedding dimension (default 32)
        month_project_dim: Month projection dimension (0 = raw 2D cyclic)
    """

    def __init__(
        self,
        visual_dim: int = 1440,
        zone_embed_dim: int = 32,
        month_project_dim: int = 0,
    ):
        super().__init__()
        self.zone_encoder = BiogeographicZoneEncoder(embed_dim=zone_embed_dim)
        self.month_encoder = CyclicMonthEncoder(project_dim=month_project_dim)
        self.output_dim = visual_dim + zone_embed_dim + self.month_encoder.output_dim

        # Layer norm on the fused vector for training stability
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(
        self,
        visual_features: torch.Tensor,
        zone_idx: torch.Tensor,
        month_enc: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visual_features: (B, visual_dim) from backbone/MLFI
            zone_idx: (B,) zone indices
            month_enc: (B, 2) cyclic month encoding
        Returns:
            (B, output_dim) fused feature vector
        """
        zone_emb = self.zone_encoder(zone_idx)       # (B, 32)
        month_emb = self.month_encoder(month_enc)     # (B, 2 or project_dim)
        fused = torch.cat([visual_features, zone_emb, month_emb], dim=1)
        return self.norm(fused)
