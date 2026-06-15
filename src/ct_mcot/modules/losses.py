from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class LossWeights:
    answer: float = 1.0
    rationale: float = 0.0
    superposition: float = 0.0
    grounding: float = 0.0
    smooth: float = 1e-4
    tau: float = 1e-4


def answer_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def rationale_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=ignore_index)


def superposition_loss(projected: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    sq = (projected - targets).pow(2).sum(dim=-1)
    if mask is not None:
        sq = sq * mask.float()
        return sq.sum() / mask.float().sum().clamp_min(1.0)
    return sq.mean()


def grounding_kl(attention: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    eps = 1e-8
    log_attn = (attention + eps).log()
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)
    kl = F.kl_div(log_attn, target, reduction="none").sum(dim=-1)
    if mask is not None:
        kl = kl * mask.float()
        return kl.sum() / mask.float().sum().clamp_min(1.0)
    return kl.mean()


def trajectory_smoothness(states: list[torch.Tensor]) -> torch.Tensor:
    if len(states) < 2:
        return states[0].new_tensor(0.0)
    diffs = [(states[i] - states[i - 1]).pow(2).mean() for i in range(1, len(states))]
    return torch.stack(diffs).mean()


def tau_regularization(tau: torch.Tensor, tau_min: float, tau_max: float) -> torch.Tensor:
    low = (tau_min - tau).clamp_min(0.0).pow(2)
    high = (tau - tau_max).clamp_min(0.0).pow(2)
    return (low + high).mean()


def weighted_sum(losses: dict[str, torch.Tensor], weights: LossWeights) -> torch.Tensor:
    total = None
    for name, loss in losses.items():
        weight = getattr(weights, name, 0.0)
        if weight == 0:
            continue
        total = loss * weight if total is None else total + loss * weight
    if total is None:
        raise ValueError("No active losses were provided.")
    return total
