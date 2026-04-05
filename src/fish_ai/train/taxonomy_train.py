"""
Training loop for the multi-head taxonomy classifier.

Source (transfer learning patterns): https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fish_ai.eval.taxonomy_metrics import classification_metrics, unknown_curve_from_logits


@dataclass(frozen=True)
class TrainConfig:
    num_epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    num_workers: int = 2


def build_label_maps(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, int]]:
    """
    Build string->id maps for family/genus/species.
    """
    out: Dict[str, Dict[str, int]] = {}
    for level in ["family", "genus", "species"]:
        vals = sorted({r[level] for r in rows})
        out[level] = {v: i for i, v in enumerate(vals)}
    return out


def encode_targets(batch_targets: List[Dict[str, str]], maps: Dict[str, Dict[str, int]], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        lvl: torch.tensor([maps[lvl][t[lvl]] for t in batch_targets], dtype=torch.int64, device=device)
        for lvl in ["family", "genus", "species"]
    }


def _collated_targets_to_list(y_str: object) -> List[Dict[str, str]]:
    """
    DataLoader default-collate for dicts returns a dict-of-lists.
    Convert both dict-of-lists and list-of-dicts into a list-of-dicts.
    """
    if isinstance(y_str, dict):
        keys = list(y_str.keys())
        n = len(y_str[keys[0]]) if keys else 0
        return [{k: y_str[k][i] for k in keys} for i in range(n)]
    return list(y_str)  # type: ignore[arg-type]


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    maps: Dict[str, Dict[str, int]],
    loss_weights: Dict[str, float] | None = None,
) -> Dict[str, float]:
    model.train()
    ce = nn.CrossEntropyLoss()
    loss_weights = loss_weights or {"family": 1.0, "genus": 1.0, "species": 1.0}

    total = 0.0
    n = 0
    parts = {"family": 0.0, "genus": 0.0, "species": 0.0}
    correct = {"family": 0, "genus": 0, "species": 0}
    n_tokens = 0

    for x, y_str in loader:
        x = x.to(device)
        y_batch = _collated_targets_to_list(y_str)
        y = encode_targets(y_batch, maps, device)

        out = model(x)
        loss_f = ce(out["family"], y["family"]) * loss_weights["family"]
        loss_g = ce(out["genus"], y["genus"]) * loss_weights["genus"]
        loss_s = ce(out["species"], y["species"]) * loss_weights["species"]
        loss = loss_f + loss_g + loss_s

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        parts["family"] += float(loss_f.detach().cpu())
        parts["genus"] += float(loss_g.detach().cpu())
        parts["species"] += float(loss_s.detach().cpu())
        total += float(loss.detach().cpu())
        n += 1
        bs = x.shape[0]
        n_tokens += bs
        for lvl in ["family", "genus", "species"]:
            pred = out[lvl].argmax(dim=1)
            correct[lvl] += int((pred == y[lvl]).sum().item())

    if n == 0:
        return {
            "loss_total": 0.0,
            "loss_family": 0.0,
            "loss_genus": 0.0,
            "loss_species": 0.0,
            "train_family_acc_top1": 0.0,
            "train_genus_acc_top1": 0.0,
            "train_species_acc_top1": 0.0,
        }

    return {
        "loss_total": total / n,
        "loss_family": parts["family"] / n,
        "loss_genus": parts["genus"] / n,
        "loss_species": parts["species"] / n,
        "train_family_acc_top1": correct["family"] / n_tokens,
        "train_genus_acc_top1": correct["genus"] / n_tokens,
        "train_species_acc_top1": correct["species"] / n_tokens,
    }


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    maps: Dict[str, Dict[str, int]],
    *,
    loss_weights: Dict[str, float] | None = None,
) -> Dict[str, Dict[str, float] | Dict[str, List[float]]]:
    model.eval()
    loss_weights = loss_weights or {"family": 1.0, "genus": 1.0, "species": 1.0}
    ce = nn.CrossEntropyLoss(reduction="sum")
    logits: Dict[str, List[np.ndarray]] = {"family": [], "genus": [], "species": []}
    ytrue: Dict[str, List[np.ndarray]] = {"family": [], "genus": [], "species": []}
    loss_sum = 0.0
    n_samples = 0

    for x, y_str in loader:
        x = x.to(device)
        y_batch = _collated_targets_to_list(y_str)
        y = encode_targets(y_batch, maps, device=device)

        out = model(x)
        lf = ce(out["family"], y["family"]) * loss_weights["family"]
        lg = ce(out["genus"], y["genus"]) * loss_weights["genus"]
        ls = ce(out["species"], y["species"]) * loss_weights["species"]
        loss_sum += float((lf + lg + ls).detach().cpu())
        n_samples += int(x.shape[0])

        for lvl in ["family", "genus", "species"]:
            logits[lvl].append(out[lvl].detach().cpu().numpy())
            ytrue[lvl].append(y[lvl].detach().cpu().numpy())

    results: Dict[str, Dict[str, float] | Dict[str, List[float]] | float] = {}
    results["loss_total"] = float(loss_sum / max(n_samples, 1))

    for lvl in ["family", "genus", "species"]:
        L = np.concatenate(logits[lvl], axis=0) if logits[lvl] else np.zeros((0, 1), dtype=np.float32)
        Y = np.concatenate(ytrue[lvl], axis=0) if ytrue[lvl] else np.zeros((0,), dtype=np.int64)
        results[lvl] = classification_metrics(L, Y)
        if lvl == "species":
            thresholds = [float(x) for x in np.linspace(0.0, 0.95, 20)]
            results["unknown_curve_species"] = unknown_curve_from_logits(L, Y, thresholds)

    return results  # type: ignore[return-value]

