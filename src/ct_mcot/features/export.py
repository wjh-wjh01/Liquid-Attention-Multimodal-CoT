from __future__ import annotations

import json
from pathlib import Path

import torch

from .cache import FeatureCache


def export_manifest_features(
    manifest_path: str | Path,
    cache_root: str | Path,
    text_dim: int = 128,
    vision_dim: int = 128,
    knowledge_dim: int = 128,
    max_text_tokens: int = 32,
    max_vision_tokens: int = 16,
    max_knowledge_tokens: int = 8,
    seed: int = 13,
) -> Path:
    """Deterministic placeholder feature exporter for pipeline dry-runs.

    Full experiments should replace this with VLM-backed feature extraction.
    """
    generator = torch.Generator().manual_seed(seed)
    cache = FeatureCache(cache_root)
    with Path(manifest_path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = _deterministic_tokens(row.get("question", ""), max_text_tokens, text_dim, generator)
            vision = _deterministic_tokens(row.get("image_path", "") or "no_image", max_vision_tokens, vision_dim, generator)
            knowledge = _deterministic_tokens(" ".join(row.get("choices", [])), max_knowledge_tokens, knowledge_dim, generator)
            features = torch.cat([text, vision, knowledge], dim=0)
            mask = torch.ones(features.shape[0], dtype=torch.bool)
            cache.put(
                example_id=str(row["id"]),
                features=features,
                mask=mask,
                label=row["label"],
                source={"manifest": str(manifest_path), "feature_mode": "deterministic_dry_run"},
                metadata={"benchmark": row.get("benchmark"), "image_path": row.get("image_path")},
            )
    return cache.manifest_path


def _deterministic_tokens(text: str, tokens: int, dim: int, generator: torch.Generator) -> torch.Tensor:
    base = torch.randn(tokens, dim, generator=generator) * 0.05
    if text:
        codepoints = torch.tensor([ord(ch) % 251 for ch in text[:tokens]], dtype=torch.float32)
        base[: codepoints.numel(), 0] += codepoints / 251.0
    return base
