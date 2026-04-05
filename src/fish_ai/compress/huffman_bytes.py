"""
Lossless Huffman coding over raw bytes (e.g. torch.save blobs).

Stage 3: smaller checkpoints on disk. Does not speed up forward() on GPU/CPU.
"""

from __future__ import annotations

import heapq
import struct
from typing import Dict, List, Tuple

MAGIC = b"FISHQC1\0"


def _build_tree(counts: List[int]) -> object | None:
    heap: List[Tuple[int, int, object]] = []
    uid = 0
    for sym in range(256):
        c = counts[sym]
        if c > 0:
            heapq.heappush(heap, (c, uid, ("leaf", sym)))
            uid += 1
    if not heap:
        return None
    if len(heap) == 1:
        return ("leaf", heap[0][2][1])
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        heapq.heappush(heap, (a[0] + b[0], uid, ("node", a[2], b[2])))
        uid += 1
    return heap[0][2]


def _build_codes(node: object, prefix: str = "") -> Dict[int, str]:
    if node is None:
        return {}
    if node[0] == "leaf":
        sym = node[1]
        return {sym: prefix if prefix else "0"}
    _, left, right = node  # type: ignore[misc]
    out: Dict[int, str] = {}
    out.update(_build_codes(left, prefix + "0"))
    out.update(_build_codes(right, prefix + "1"))
    return out


def _pack_bits(bits: str) -> Tuple[bytes, int]:
    pad = (-len(bits)) % 8
    bits_padded = bits + ("0" * pad)
    out = bytearray()
    for i in range(0, len(bits_padded), 8):
        out.append(int(bits_padded[i : i + 8], 2))
    return bytes(out), pad


def _bits_from_bytes(data: bytes, num_bits: int) -> str:
    bits = "".join(f"{b:08b}" for b in data)
    return bits[:num_bits]


def _decode_bits(bits: str, root: object) -> bytearray:
    out = bytearray()
    if root[0] == "leaf":
        sym = root[1]
        return bytearray([sym] * len(bits))

    node = root
    for bit in bits:
        if node[0] != "node":
            break
        _, left, right = node  # type: ignore[misc]
        node = left if bit == "0" else right
        if node[0] == "leaf":
            out.append(node[1])
            node = root
    return out


def huffman_encode(data: bytes) -> bytes:
    if not data:
        header = struct.pack(">256I", *([0] * 256))
        return MAGIC + header + struct.pack(">Q", 0) + struct.pack(">B", 0) + b""

    counts = [0] * 256
    for b in data:
        counts[b] += 1

    root = _build_tree(counts)
    if root is None:
        header = struct.pack(">256I", *counts)
        return MAGIC + header + struct.pack(">Q", 0) + struct.pack(">B", 0) + b""

    codes = _build_codes(root)
    bits = "".join(codes[b] for b in data)
    packed, pad = _pack_bits(bits)
    header = struct.pack(">256I", *counts)
    meta = struct.pack(">QB", len(bits), pad)
    return MAGIC + header + meta + packed


def huffman_decode(blob: bytes) -> bytes:
    if not blob.startswith(MAGIC):
        raise ValueError("Not a FISHQC1 Huffman payload")
    off = len(MAGIC)
    counts = list(struct.unpack_from(">256I", blob, off))
    off += 256 * 4
    num_bits, pad = struct.unpack_from(">QB", blob, off)
    off += 8 + 1
    packed = blob[off:]

    if num_bits == 0:
        return b""

    root = _build_tree(counts)
    if root is None:
        return b""

    bits = _bits_from_bytes(packed, num_bits)
    return bytes(_decode_bits(bits, root))
