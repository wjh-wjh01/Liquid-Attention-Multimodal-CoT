from __future__ import annotations

import torch


def attention_entropy(attention: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return -(attention * (attention + eps).log()).sum(dim=-1)


def top_attention_tokens(attention: torch.Tensor, k: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    pooled = attention.mean(dim=1)
    return torch.topk(pooled, k=min(k, pooled.shape[-1]), dim=-1)


def modality_attention_mass(attention: torch.Tensor, modality: torch.Tensor, num_modalities: int = 3) -> torch.Tensor:
    pooled = attention.mean(dim=1)
    masses = []
    for idx in range(num_modalities):
        masses.append((pooled * (modality == idx).float()).sum(dim=-1))
    return torch.stack(masses, dim=-1)
