"""
Minimal FedAvg implementation for simulated federated learning (pure PyTorch).

We keep this lightweight and device-flexible (CPU/CUDA/MPS).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch


@dataclass(frozen=True)
class FedAvgConfig:
    num_clients: int = 10
    num_rounds: int = 20
    clients_per_round: int | None = None  # None => all clients
    local_epochs: int = 1
    seed: int = 42


def split_indices_stratified(
    labels: Sequence[int],
    num_clients: int,
    seed: int = 42,
) -> List[List[int]]:
    """
    Stratified-ish split: shuffle indices within each class then round-robin assign to clients.
    """
    import numpy as np

    rng = np.random.RandomState(seed)
    labels = list(labels)
    clients: List[List[int]] = [[] for _ in range(num_clients)]
    by_class: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        by_class.setdefault(int(y), []).append(i)
    for cls, idxs in by_class.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        for j, idx in enumerate(idxs):
            clients[j % num_clients].append(idx)
    for c in clients:
        rng.shuffle(c)
    return clients


def average_state_dicts(state_dicts: List[Dict[str, torch.Tensor]], weights: List[float] | None = None) -> Dict[str, torch.Tensor]:
    """
    Weighted average of model parameters (FedAvg).
    """
    if not state_dicts:
        raise ValueError("No state_dicts to average")
    if weights is None:
        weights = [1.0 for _ in state_dicts]
    wsum = float(sum(weights))
    if wsum <= 0:
        raise ValueError("Sum of weights must be > 0")

    avg: Dict[str, torch.Tensor] = {}
    for k in state_dicts[0].keys():
        tensors = [sd[k].detach().float().cpu() * (weights[i] / wsum) for i, sd in enumerate(state_dicts)]
        avg[k] = torch.stack(tensors, dim=0).sum(dim=0)
    return avg


def select_clients(num_clients: int, clients_per_round: int | None, round_idx: int, seed: int = 42) -> List[int]:
    import numpy as np

    rng = np.random.RandomState(seed + round_idx)
    if clients_per_round is None or clients_per_round >= num_clients:
        return list(range(num_clients))
    return rng.choice(np.arange(num_clients), size=int(clients_per_round), replace=False).tolist()

