from __future__ import annotations

import torch


def grid_laplacian(height: int, width: int, normalized: bool = True, device: torch.device | None = None) -> torch.Tensor:
    n = height * width
    adj = torch.zeros(n, n, device=device)
    for y in range(height):
        for x in range(width):
            idx = y * width + x
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < height and 0 <= xx < width:
                    adj[idx, yy * width + xx] = 1.0
    degree = adj.sum(dim=-1)
    lap = torch.diag(degree) - adj
    if not normalized:
        return lap
    inv_sqrt = degree.clamp_min(1.0).pow(-0.5)
    return inv_sqrt[:, None] * lap * inv_sqrt[None, :]


def relation_bias(relation_ids: torch.Tensor, embedding: torch.nn.Embedding) -> torch.Tensor:
    """Map relation ids shaped [batch, slots, tokens] to scalar attention biases."""
    return embedding(relation_ids).squeeze(-1)


def make_modality_mask(modality: torch.Tensor, allowed: list[int] | None) -> torch.Tensor:
    if allowed is None:
        return torch.ones_like(modality, dtype=torch.bool)
    mask = torch.zeros_like(modality, dtype=torch.bool)
    for item in allowed:
        mask |= modality == item
    return mask
