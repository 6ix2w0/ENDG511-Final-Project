"""
Utilities for building few-shot splits from taxonomy manifests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from fish_ai.data.jsonl import read_jsonl, write_jsonl


@dataclass(frozen=True)
class FewShotConfig:
    k: int
    seed: int = 42
    min_images_per_species: int = 0


def load_taxonomy_manifest(path: str | Path) -> List[Dict[str, Any]]:
    return list(read_jsonl(path))


def build_kshot_train_rows(
    train_rows: List[Dict[str, Any]],
    *,
    k: int,
    seed: int,
    top_n_species: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Select exactly k samples per species from the provided rows.
    If a species has < k samples, it is dropped.
    """
    df = pd.DataFrame(
        {
            "idx": list(range(len(train_rows))),
            "species": [r["taxonomy"]["species"] for r in train_rows],
        }
    )
    vc = df["species"].value_counts()
    eligible = vc[vc >= k].index.tolist()
    if top_n_species is not None:
        eligible = eligible[:top_n_species]

    rng = np.random.RandomState(seed)
    chosen_idx: List[int] = []
    for sp in eligible:
        sp_idx = df.loc[df["species"] == sp, "idx"].to_numpy()
        picked = rng.choice(sp_idx, size=k, replace=False)
        chosen_idx.extend(picked.tolist())

    chosen_idx = sorted(chosen_idx)
    return [train_rows[i] for i in chosen_idx]


def write_fewshot_train_manifest(
    in_train_manifest: str | Path,
    out_manifest: str | Path,
    *,
    k: int,
    seed: int,
    top_n_species: Optional[int] = None,
) -> Path:
    rows = load_taxonomy_manifest(in_train_manifest)
    fewshot = build_kshot_train_rows(rows, k=k, seed=seed, top_n_species=top_n_species)
    out_manifest = Path(out_manifest)
    write_jsonl(out_manifest, fewshot)
    return out_manifest

