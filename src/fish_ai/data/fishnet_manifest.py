"""
Build taxonomy manifests from FishNet annotation CSVs.

Source (dataset/project):
https://fishnet-2023.github.io/
https://github.com/faixan-khan/FishNet

FishNet provides annotation CSVs in the GitHub repo under `anns/`:
- train.csv
- test.csv

The actual image archive is downloaded separately (Google Drive link on project page).
This script is written to be flexible about image folder layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd

from fish_ai.data.jsonl import write_jsonl


@dataclass(frozen=True)
class FishNetLayout:
    """
    Where FishNet files live locally.

    Expected minimal inputs:
    - train_csv / test_csv from FishNet `anns/`
    - images_root: directory containing the image files (may be flat or nested)
    """

    train_csv: Path
    test_csv: Path
    images_root: Path


def _url_basename(url: str) -> str:
    """Last path segment of a URL, or empty if missing."""
    path = urlparse(url.strip()).path or ""
    name = Path(path).name
    return name


def _resolve_image_path(images_root: Path, row: pd.Series) -> Optional[Path]:
    """
    FishNet CSV includes an `image` field (filename) and often a `Folder` field.

    Some rows store a full ``https://...`` URL in ``image`` (FishBase, etc.). Those are not
    local paths: we only keep a row if a file exists under ``images_root``. For URLs we try
    the same layouts using the URL's **basename** (e.g. ``Vapue_u3.jpg``).

    Returns ``None`` when no existing local file is found (row should be skipped).
    """
    raw = str(row.get("image", "") or "").strip()
    if not raw:
        return None

    fname = raw
    if raw.lower().startswith(("http://", "https://")):
        fname = _url_basename(raw)
        if not fname:
            return None

    folder = str(row.get("Folder", "") or "")
    source = str(row.get("source", "") or "")

    candidates: List[Optional[Path]] = [
        images_root / fname,
        images_root / folder / fname if folder else None,
        images_root / source / folder / fname if source and folder else None,
        images_root / source / fname if source else None,
    ]
    for c in candidates:
        if c is not None and c.is_file():
            return c.resolve()

    return None


def _taxonomy_from_row(row: pd.Series) -> Dict[str, str]:
    # FishNet CSV has Family / Genus and sometimes species string column appears blank.
    # The `species` column exists in test.csv but may be empty. Use Genus + SpecCode when needed.
    family = str(row.get("Family", "") or "").strip()
    genus = str(row.get("Genus", "") or "").strip()

    species = str(row.get("species", "") or "").strip()
    if not species:
        # Try to build a stable species string
        speccode = str(row.get("SpecCode", "") or "").strip()
        species = f"{genus}__{speccode}" if speccode else genus
    return {"family": family or "UnknownFamily", "genus": genus or "UnknownGenus", "species": species or "UnknownSpecies"}


def build_taxonomy_rows_from_csv(
    csv_path: Path,
    images_root: Path,
    split: str,
) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    rows: List[Dict[str, Any]] = []
    for i, r in df.iterrows():
        img_path = _resolve_image_path(images_root, r)
        if img_path is None:
            continue
        tax = _taxonomy_from_row(r)
        rows.append(
            {
                "sample_id": str(r.get("Unnamed: 0", i)),
                "image_path": str(img_path),
                "split": split,
                "taxonomy": tax,
                "source_dataset": "FishNet",
                "source_uri": "https://fishnet-2023.github.io/",
            }
        )
    return rows


def filter_top_species(
    train_rows: List[Dict[str, Any]],
    top_n: int = 100,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    species = [r["taxonomy"]["species"] for r in train_rows]
    vc = pd.Series(species).value_counts()
    keep = set(vc.head(top_n).index.tolist())
    filtered = [r for r in train_rows if r["taxonomy"]["species"] in keep]
    return filtered, list(keep)


def write_fishnet_taxonomy_manifests(
    layout: FishNetLayout,
    out_dir: Path,
    *,
    top_n_species: int = 100,
    val_frac_from_train: float = 0.1,
    seed: int = 42,
) -> Dict[str, Path]:
    """
    FishNet provides train/test. We create val by splitting train.

    Rows whose ``image`` field does not resolve to an existing file under ``images_root``
    (including URL-only references with no matching local basename) are omitted.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = build_taxonomy_rows_from_csv(layout.train_csv, layout.images_root, split="train")
    test_rows = build_taxonomy_rows_from_csv(layout.test_csv, layout.images_root, split="test")

    train_rows, keep_species = filter_top_species(train_rows, top_n=top_n_species)
    test_rows = [r for r in test_rows if r["taxonomy"]["species"] in set(keep_species)]

    # Deterministic val split
    rng = pd.Series(range(len(train_rows))).sample(frac=1.0, random_state=seed).tolist()
    n_val = int(len(train_rows) * val_frac_from_train)
    val_idx = set(rng[:n_val])

    train_final = []
    val_final = []
    for i, r in enumerate(train_rows):
        if i in val_idx:
            r = dict(r)
            r["split"] = "val"
            val_final.append(r)
        else:
            train_final.append(r)

    paths = {
        "train": out_dir / "fishnet_taxonomy_train.jsonl",
        "val": out_dir / "fishnet_taxonomy_val.jsonl",
        "test": out_dir / "fishnet_taxonomy_test.jsonl",
    }
    write_jsonl(paths["train"], train_final)
    write_jsonl(paths["val"], val_final)
    write_jsonl(paths["test"], test_rows)
    return paths

