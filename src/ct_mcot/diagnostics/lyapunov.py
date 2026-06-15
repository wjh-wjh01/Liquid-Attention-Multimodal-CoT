from __future__ import annotations

from collections.abc import Callable

import torch


@torch.no_grad()
def finite_difference_sensitivity(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    memory: torch.Tensor,
    epsilon: float = 1e-3,
    max_tokens: int | None = None,
) -> torch.Tensor:
    """Token-level terminal-state sensitivity by finite differences."""
    base = forward_fn(memory)
    token_count = memory.shape[1] if max_tokens is None else min(max_tokens, memory.shape[1])
    scores = []
    for idx in range(token_count):
        perturb = torch.zeros_like(memory)
        perturb[:, idx, :] = epsilon
        changed = forward_fn(memory + perturb)
        scores.append((changed - base).flatten(1).norm(dim=-1) / epsilon)
    return torch.stack(scores, dim=-1)


def finite_time_lyapunov(delta0: torch.Tensor, delta_t: torch.Tensor, horizon: float) -> torch.Tensor:
    return (delta_t.norm(dim=-1).clamp_min(1e-12) / delta0.norm(dim=-1).clamp_min(1e-12)).log() / horizon
