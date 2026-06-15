from __future__ import annotations

import torch
from torch import nn


class AnswerHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, thought: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        thought_pool = thought.mean(dim=1)
        if mask is None:
            memory_pool = memory.mean(dim=1)
        else:
            weights = mask.float().unsqueeze(-1)
            memory_pool = (memory * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.net(torch.cat([thought_pool, memory_pool], dim=-1))


class CandidateScorer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.answer = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim**0.5

    def forward(self, thought: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        q = self.query(thought.mean(dim=1)).unsqueeze(1)
        a = self.answer(candidates)
        return (q * a).sum(dim=-1) / self.scale


class RationaleDecoder(nn.Module):
    """Small GRU rationale head for artifact completeness.

    This is intended for supervised/teacher-rationale experiments. For large
    language-model decoding, connect the terminal thought state to the LLM
    adapter instead of using this compact head.
    """

    def __init__(self, hidden_dim: int, vocab_size: int, max_len: int = 64):
        super().__init__()
        self.max_len = max_len
        self.bos = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, thought: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        batch = thought.shape[0]
        initial = thought.mean(dim=1).unsqueeze(0)
        if targets is None:
            inputs = self.bos.expand(batch, self.max_len, -1)
        else:
            inputs = self.bos.expand(batch, targets.shape[1], -1)
        hidden, _ = self.gru(inputs, initial)
        return self.out(hidden)
