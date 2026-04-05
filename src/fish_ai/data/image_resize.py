"""
Shared PIL resize helpers for faster training (smaller tensors before the model).
"""

from __future__ import annotations

from PIL import Image


def uniform_scale_shorter_max(
    width: int,
    height: int,
    *,
    shorter_side: int | None = None,
    max_side: int | None = None,
) -> float:
    """
    TorchVision-style scale: first match ``shorter_side`` on min(w,h), then clamp long edge to ``max_side``.
    If only ``max_side`` is set, scale down when max(w,h) exceeds it. Returns 1.0 when no resize applies.
    """
    if shorter_side is None and max_side is None:
        return 1.0
    if width <= 0 or height <= 0:
        return 1.0

    im_min = min(width, height)
    im_max = max(width, height)
    scale = 1.0
    if shorter_side is not None:
        scale = shorter_side / im_min
    tw = width * scale
    th = height * scale
    cur_max = max(tw, th)
    if max_side is not None and cur_max > max_side:
        scale *= max_side / cur_max
    return float(scale)


def uniform_scale_cap_max_edge(width: int, height: int, max_side: int) -> float:
    """Scale down only if max(width, height) > max_side (preserves aspect ratio)."""
    if max_side <= 0:
        return 1.0
    m = max(width, height)
    if m <= max_side:
        return 1.0
    return float(max_side / m)


def resize_pil_bilinear(img: Image.Image, scale: float) -> Image.Image:
    if scale == 1.0:
        return img
    w, h = img.size
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return img.resize((nw, nh), Image.BILINEAR)
