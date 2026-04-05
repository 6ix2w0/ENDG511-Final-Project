"""
Dataset for hierarchical fish taxonomy classification from JSONL manifests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from fish_ai.data.image_resize import resize_pil_bilinear, uniform_scale_cap_max_edge
from fish_ai.data.jsonl import read_jsonl


@dataclass(frozen=True)
class TaxonomyRow:
    sample_id: str
    image_path: str
    family: str
    genus: str
    species: str


class FishTaxonomyDataset(Dataset):
    """
    Returns (image, target) where target contains string taxonomy labels.
    The training code will map strings to integer ids.

    ``max_side_before_square`` optionally downscales the PIL image (longest edge cap) before the
    square ``Resize((image_size, image_size))``, which speeds IO+transform on very large crops.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        image_size: int = 224,
        augment: bool = False,
        *,
        max_side_before_square: int | None = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.max_side_before_square = max_side_before_square
        self.rows: List[TaxonomyRow] = []
        for r in read_jsonl(self.manifest_path):
            tax = r.get("taxonomy") or {}
            self.rows.append(
                TaxonomyRow(
                    sample_id=str(r.get("sample_id") or r.get("image_id") or r.get("id")),
                    image_path=str(r["image_path"]),
                    family=str(tax["family"]),
                    genus=str(tax["genus"]),
                    species=str(tax["species"]),
                )
            )

        base = [
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
        ]
        if self.augment:
            aug = [
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
                T.RandomHorizontalFlip(p=0.5),
            ]
            self.tf = T.Compose(aug + base)
        else:
            self.tf = T.Compose(base)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, str]]:
        row = self.rows[idx]
        img = Image.open(row.image_path).convert("RGB")
        if self.max_side_before_square is not None:
            w, h = img.size
            s = uniform_scale_cap_max_edge(w, h, self.max_side_before_square)
            img = resize_pil_bilinear(img, s)
        x = self.tf(img)
        y = {"family": row.family, "genus": row.genus, "species": row.species}
        return x, y

