from __future__ import annotations

import torch
from torch import nn


class LatentRecurrentModel(nn.Module):
    """MCOUT/COCONUT-style latent recurrent baseline without dynamic attention logits."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        thought_slots: int = 8,
        recurrent_steps: int = 12,
    ):
        super().__init__()
        self.thought_slots = thought_slots
        self.recurrent_steps = recurrent_steps
        self.memory_proj = nn.Linear(input_dim, hidden_dim)
        self.initial_slots = nn.Parameter(torch.randn(thought_slots, hidden_dim) * 0.02)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.gru = nn.GRUCell(2 * hidden_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, memory: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        memory = self.memory_proj(memory)
        batch = memory.shape[0]
        thought = self.initial_slots.unsqueeze(0).expand(batch, -1, -1)
        trajectory = []
        for _ in range(self.recurrent_steps):
            q = self.query(thought)
            k = self.key(memory)
            logits = torch.matmul(q, k.transpose(1, 2)) / (memory.shape[-1] ** 0.5)
            if mask is not None:
                logits = logits.masked_fill(~mask.unsqueeze(1), -1e9)
            evidence = torch.matmul(torch.softmax(logits, dim=-1), self.value(memory))
            gru_input = torch.cat([thought, evidence], dim=-1).reshape(batch * self.thought_slots, -1)
            thought = self.gru(gru_input, thought.reshape(batch * self.thought_slots, -1))
            thought = thought.reshape(batch, self.thought_slots, -1)
            trajectory.append(thought)
        mem_pool = _masked_mean(memory, mask)
        thought_pool = thought.mean(dim=1)
        return {"logits": self.head(torch.cat([thought_pool, mem_pool], dim=-1)), "trajectory": trajectory}


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    weights = mask.float().unsqueeze(-1)
    return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
