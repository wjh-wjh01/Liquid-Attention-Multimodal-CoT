from __future__ import annotations

import pathlib
from typing import Dict, Optional

import numpy as np
from PIL import Image


def _pad_or_trim(vec: np.ndarray, dim: int) -> np.ndarray:
    if vec.size == dim:
        return vec.astype(np.float32)
    if vec.size > dim:
        return vec[:dim].astype(np.float32)
    out = np.zeros((dim,), dtype=np.float32)
    out[: vec.size] = vec
    return out


def image_to_feature(path: Optional[str], dim: int, cache: Dict[str, np.ndarray]) -> np.ndarray:
    if path is None:
        return np.zeros((dim,), dtype=np.float32)
    if path in cache:
        return cache[path]

    p = pathlib.Path(path)
    if not p.exists():
        feat = np.zeros((dim,), dtype=np.float32)
        cache[path] = feat
        return feat

    try:
        img = Image.open(p).convert("RGB").resize((64, 64))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        mean = arr.mean(axis=(0, 1))
        std = arr.std(axis=(0, 1))
        mn = arr.min(axis=(0, 1))
        mx = arr.max(axis=(0, 1))
        stats = np.concatenate([mean, std, mn, mx], axis=0)

        hist_parts = []
        for c in range(3):
            hist, _ = np.histogram(arr[:, :, c], bins=8, range=(0.0, 1.0), density=True)
            hist_parts.append(hist)
        hist_vec = np.concatenate(hist_parts, axis=0)

        feat = np.concatenate([stats, hist_vec], axis=0).astype(np.float32)
        feat = _pad_or_trim(feat, dim)
    except Exception:
        feat = np.zeros((dim,), dtype=np.float32)

    cache[path] = feat
    return feat
