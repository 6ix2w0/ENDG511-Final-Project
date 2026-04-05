"""
Three-stage taxonomy checkpoint pipeline:

1) L1 unstructured prune on classification heads (optional).
2) Dynamic INT8 quantization of Linear + Conv2d (CPU inference; main latency win on CPU).
3) Huffman coding of the serialized checkpoint bytes (disk size; not forward speed).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from fish_ai.compress.huffman_bytes import huffman_decode, huffman_encode
from fish_ai.models.taxonomy_classifier import TaxonomyClassifier, TaxonomyHeadSizes


@dataclass(frozen=True)
class TaxonomyCompressionConfig:
    """Defaults tuned for CPU inference + smaller artifacts."""

    prune_head_l1_amount: float = 0.20
    apply_prune: bool = True
    quantize: bool = True
    huffman_wrap: bool = True


def maps_to_head_sizes(maps: Dict[str, Dict[str, int]]) -> TaxonomyHeadSizes:
    return TaxonomyHeadSizes(
        n_family=len(maps["family"]),
        n_genus=len(maps["genus"]),
        n_species=len(maps["species"]),
    )


def _prune_taxonomy_heads(model: TaxonomyClassifier, amount: float) -> None:
    for name in ("head_family", "head_genus", "head_species"):
        m = getattr(model, name)
        prune.l1_unstructured(m, name="weight", amount=amount)
        prune.remove(m, "weight")


def load_float_taxonomy_from_ckpt(ckpt: Dict[str, Any]) -> TaxonomyClassifier:
    maps = ckpt["maps"]
    backbone = str(ckpt.get("backbone", "efficientnet_b0"))
    sizes = maps_to_head_sizes(maps)
    model = TaxonomyClassifier(sizes, backbone=backbone, pretrained=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model


def compress_taxonomy_checkpoint(
    src_path: str | Path,
    dst_path: str | Path,
    cfg: TaxonomyCompressionConfig = TaxonomyCompressionConfig(),
) -> Dict[str, Any]:
    """
    Load float taxonomy .pt, apply prune → quantize, optionally Huffman-wrap, write ``dst_path``.

    Returns a small report dict (byte sizes).
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)

    ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
    model = load_float_taxonomy_from_ckpt(ckpt)
    model.cpu()

    if cfg.apply_prune and cfg.prune_head_l1_amount > 0:
        _prune_taxonomy_heads(model, cfg.prune_head_l1_amount)

    if cfg.quantize:
        model = torch.ao.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8,
            inplace=False,
        )

    out_ckpt: Dict[str, Any] = {
        "model_state": model.state_dict(),
        "backbone": ckpt.get("backbone", "efficientnet_b0"),
        "maps": ckpt["maps"],
        "compression": {
            "pipeline": "prune_l1_heads+dynamic_qint8+huffman_optional",
            "prune_head_l1_amount": cfg.prune_head_l1_amount if cfg.apply_prune else 0.0,
            "quantized": cfg.quantize,
            "huffman": cfg.huffman_wrap,
        },
    }
    if "cfg" in ckpt:
        out_ckpt["cfg"] = ckpt["cfg"]

    buf = io.BytesIO()
    torch.save(out_ckpt, buf)
    raw = buf.getvalue()
    report: Dict[str, Any] = {"raw_torch_bytes": len(raw)}

    if cfg.huffman_wrap:
        wrapped = huffman_encode(raw)
        dst_path.write_bytes(wrapped)
        report["file_bytes"] = len(wrapped)
    else:
        dst_path.write_bytes(raw)
        report["file_bytes"] = len(raw)

    report["src_bytes"] = src_path.stat().st_size
    return report


def load_taxonomy_checkpoint_auto(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    """Load .pt whether plain torch or Huffman-wrapped."""
    path = Path(path)
    blob = path.read_bytes()
    if blob.startswith(b"FISHQC1"):
        blob = huffman_decode(blob)
    return torch.load(io.BytesIO(blob), map_location=map_location, weights_only=False)


def build_quantized_taxonomy_for_inference(ckpt: Dict[str, Any]) -> TaxonomyClassifier:
    """
    Rebuild quantized model (must match ``compress_taxonomy_checkpoint`` quant settings).
    """
    maps = ckpt["maps"]
    backbone = str(ckpt.get("backbone", "efficientnet_b0"))
    sizes = maps_to_head_sizes(maps)
    float_m = TaxonomyClassifier(sizes, backbone=backbone, pretrained=False)
    q_m = torch.ao.quantization.quantize_dynamic(
        float_m,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8,
        inplace=False,
    )
    q_m.load_state_dict(ckpt["model_state"], strict=True)
    q_m.eval()
    return q_m


def load_taxonomy_for_inference(ckpt: Dict[str, Any]) -> TaxonomyClassifier:
    """Load float or quantized taxonomy checkpoint for eval."""
    comp = ckpt.get("compression") or {}
    if comp.get("quantized"):
        return build_quantized_taxonomy_for_inference(ckpt)
    return load_float_taxonomy_from_ckpt(ckpt)
