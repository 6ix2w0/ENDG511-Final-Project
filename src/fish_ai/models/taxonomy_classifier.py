"""
Multi-head taxonomy classifier (family/genus/species).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torchvision.models as tvm


@dataclass(frozen=True)
class TaxonomyHeadSizes:
    n_family: int
    n_genus: int
    n_species: int


class TaxonomyClassifier(nn.Module):
    def __init__(self, sizes: TaxonomyHeadSizes, backbone: str = "efficientnet_b0", pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone
        self.pretrained = pretrained

        if backbone == "efficientnet_b0":
            weights = tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            net = tvm.efficientnet_b0(weights=weights)
            feat_dim = net.classifier[1].in_features
            net.classifier = nn.Identity()
            self.backbone = net
        elif backbone == "resnet50":
            weights = tvm.ResNet50_Weights.DEFAULT if pretrained else None
            net = tvm.resnet50(weights=weights)
            feat_dim = net.fc.in_features
            net.fc = nn.Identity()
            self.backbone = net
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.head_family = nn.Linear(feat_dim, sizes.n_family)
        self.head_genus = nn.Linear(feat_dim, sizes.n_genus)
        self.head_species = nn.Linear(feat_dim, sizes.n_species)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.backbone(x)
        if feats.ndim > 2:
            feats = torch.flatten(feats, 1)
        return {
            "family": self.head_family(feats),
            "genus": self.head_genus(feats),
            "species": self.head_species(feats),
        }

