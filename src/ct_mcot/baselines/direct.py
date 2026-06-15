from __future__ import annotations

import torch
from torch import nn


class DirectPredictionModel(nn.Module):
    """No-CoT baseline over precomputed multimodal memory tokens."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, pooling: str = "mean"):
        super().__init__()
        if pooling not in {"mean", "attentive"}:
            raise ValueError(f"Unsupported pooling mode: {pooling}")
        self.pooling = pooling
        self.proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.attn = nn.Linear(hidden_dim, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, memory: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        hidden = self.proj(memory)
        pooled = self._pool(hidden, mask)
        return {"logits": self.head(pooled), "pooled": pooled}

    def _pool(self, hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if self.pooling == "mean":
            if mask is None:
                return hidden.mean(dim=1)
            weights = mask.float().unsqueeze(-1)
            return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        logits = self.attn(hidden).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        return torch.matmul(torch.softmax(logits, dim=-1).unsqueeze(1), hidden).squeeze(1)
