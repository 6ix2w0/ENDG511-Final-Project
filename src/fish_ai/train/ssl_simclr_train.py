"""
Training loop for SimCLR pretraining.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from fish_ai.models.simclr import SimCLRConfig, SimCLRModel, nt_xent_loss
from fish_ai.utils.device import get_best_device


@dataclass(frozen=True)
class TrainConfig:
    num_epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    num_workers: int = 2


def fit(
    dataset: Dataset,
    ssl_cfg: SimCLRConfig = SimCLRConfig(),
    cfg: TrainConfig = TrainConfig(),
    device: torch.device | None = None,
) -> Tuple[SimCLRModel, List[Dict[str, Any]]]:
    device = device or get_best_device()
    model = SimCLRModel(ssl_cfg, pretrained_backbone=True).to(device)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history: List[Dict[str, Any]] = []

    for epoch in range(cfg.num_epochs):
        model.train()
        total = 0.0
        n = 0
        for x1, x2 in loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            z1 = model(x1)
            z2 = model(x2)
            loss = nt_xent_loss(z1, z2, temperature=ssl_cfg.temperature)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += float(loss.detach().cpu())
            n += 1
        avg = total / max(n, 1)
        print(f"epoch={epoch+1}/{cfg.num_epochs} loss={avg:.4f}")
        history.append({"epoch": epoch + 1, "loss_nt_xent": avg})

    return model, history

