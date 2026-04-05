"""
Simple SimCLR-style self-supervised pretraining in pure PyTorch.

We implement:
- Backbone encoder (e.g., ResNet-50)
- Projection head (MLP)
- NT-Xent (InfoNCE) loss over 2N augmented views

Reference (conceptual): SimCLR paper (Chen et al., 2020).
This file is an original implementation (no code copied).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


@dataclass(frozen=True)
class SimCLRConfig:
    backbone: str = "resnet50"
    proj_dim: int = 128
    proj_hidden_dim: int = 2048
    temperature: float = 0.2


class SimCLRModel(nn.Module):
    def __init__(self, cfg: SimCLRConfig, pretrained_backbone: bool = True):
        super().__init__()
        self.cfg = cfg

        if cfg.backbone == "resnet50":
            weights = tvm.ResNet50_Weights.DEFAULT if pretrained_backbone else None
            net = tvm.resnet50(weights=weights)
            feat_dim = net.fc.in_features
            net.fc = nn.Identity()
            self.encoder = net
        elif cfg.backbone == "resnet18":
            weights = tvm.ResNet18_Weights.DEFAULT if pretrained_backbone else None
            net = tvm.resnet18(weights=weights)
            feat_dim = net.fc.in_features
            net.fc = nn.Identity()
            self.encoder = net
        else:
            raise ValueError(f"Unsupported backbone: {cfg.backbone}")

        self.projector = nn.Sequential(
            nn.Linear(feat_dim, cfg.proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.proj_hidden_dim, cfg.proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        if h.ndim > 2:
            h = torch.flatten(h, 1)
        z = self.projector(h)
        z = F.normalize(z, dim=1)
        return z


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """
    z1, z2: [N, D] normalized embeddings for two views.
    Returns scalar loss.
    """
    N, D = z1.shape
    z = torch.cat([z1, z2], dim=0)  # [2N, D]

    # Cosine similarity matrix
    sim = torch.mm(z, z.t()) / temperature  # [2N,2N]

    # Mask out self-similarity
    mask = torch.eye(2 * N, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    # Positive pairs: i <-> i+N
    pos = torch.cat([torch.arange(N, 2 * N, device=z.device), torch.arange(0, N, device=z.device)], dim=0)
    logits = sim
    labels = pos

    loss = F.cross_entropy(logits, labels)
    return loss

