"""
Datasets for self-supervised learning (fish images only).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from fish_ai.data.image_resize import resize_pil_bilinear, uniform_scale_cap_max_edge
from fish_ai.data.jsonl import read_jsonl


class TwoCropTransform:
    def __init__(self, base_transform: T.Compose):
        self.base_transform = base_transform

    def __call__(self, img: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.base_transform(img), self.base_transform(img)


class FishSSLManifestDataset(Dataset):
    """
    Reads a taxonomy manifest JSONL and returns two augmented views of each fish image.

    ``max_side_before_augment`` caps the longest edge before RandomResizedCrop (faster on huge JPEGs).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        image_size: int = 224,
        *,
        max_side_before_augment: int | None = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.image_size = int(image_size)
        self.max_side_before_augment = max_side_before_augment
        self.image_paths: List[str] = []
        for r in read_jsonl(self.manifest_path):
            self.image_paths.append(str(r["image_path"]))

        # SimCLR-ish augmentations (kept light for stability on MPS/CPU too)
        aug = T.Compose(
            [
                T.RandomResizedCrop(self.image_size, scale=(0.5, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
                T.RandomGrayscale(p=0.1),
                T.ToTensor(),
            ]
        )
        self.tf = TwoCropTransform(aug)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.max_side_before_augment is not None:
            w, h = img.size
            s = uniform_scale_cap_max_edge(w, h, self.max_side_before_augment)
            img = resize_pil_bilinear(img, s)
        x1, x2 = self.tf(img)
        return x1, x2

