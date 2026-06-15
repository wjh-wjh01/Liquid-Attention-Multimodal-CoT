from __future__ import annotations

import torch
from torch import nn

from ct_mcot.solvers import SolverConfig, solve


class ODETransformerBaseline(nn.Module):
    """Continuous-depth hidden-state baseline without explicit attention-logit dynamics."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_heads: int = 4,
        steps: int = 12,
    ):
        super().__init__()
        self.steps = steps
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.block = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            batch_first=True,
            activation="gelu",
        )
        self.leak = nn.Parameter(torch.tensor(1.0))
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_classes))

    def forward(self, memory: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        hidden = self.proj(memory)

        def field(_: float, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            h, dummy = state
            updated = self.block(h, src_key_padding_mask=(~mask if mask is not None else None))
            return -self.leak.abs() * h + updated, dummy * 0.0

        h, _ = solve(field, (hidden, hidden.new_zeros(1)), SolverConfig("rk4", 1.0, self.steps))
        pooled = _masked_mean(h, mask)
        return {"logits": self.head(pooled), "hidden": h}


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    weights = mask.float().unsqueeze(-1)
    return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
