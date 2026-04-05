"""
Metrics for hierarchical taxonomy classification + unknown-threshold evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def topk_accuracy(logits: np.ndarray, y_true: np.ndarray, k: int = 1) -> float:
    if logits.size == 0:
        return 0.0
    topk = np.argpartition(-logits, kth=min(k, logits.shape[1] - 1), axis=1)[:, :k]
    return float(np.mean([yt in topk[i] for i, yt in enumerate(y_true)]))


def classification_metrics(logits: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    y_pred = logits.argmax(axis=1)
    return {
        "acc_top1": float(accuracy_score(y_true, y_pred)),
        "acc_top5": float(topk_accuracy(logits, y_true, k=min(5, logits.shape[1]))),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def unknown_curve_from_logits(
    logits: np.ndarray,
    y_true: np.ndarray,
    thresholds: List[float],
) -> Dict[str, List[float]]:
    """
    Option A: predict Unknown if max softmax prob < threshold.
    Returns accuracy over non-unknown predictions vs coverage.
    """
    probs = softmax(logits, axis=1)
    conf = probs.max(axis=1)
    y_pred = probs.argmax(axis=1)

    coverages: List[float] = []
    accuracies: List[float] = []
    unknown_rates: List[float] = []

    n = len(y_true)
    for t in thresholds:
        keep = conf >= t
        coverage = float(np.mean(keep)) if n else 0.0
        if np.any(keep):
            acc = float(np.mean((y_pred[keep] == y_true[keep]).astype(np.float32)))
        else:
            acc = 0.0
        coverages.append(coverage)
        accuracies.append(acc)
        unknown_rates.append(1.0 - coverage)

    return {"threshold": thresholds, "coverage": coverages, "accuracy": accuracies, "unknown_rate": unknown_rates}

