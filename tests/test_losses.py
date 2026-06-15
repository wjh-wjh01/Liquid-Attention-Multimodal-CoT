from __future__ import annotations

import torch

from ct_mcot.modules.losses import answer_loss, grounding_kl


def test_answer_loss_scalar() -> None:
    logits = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 2, 1])
    assert answer_loss(logits, labels).ndim == 0


def test_grounding_kl_scalar() -> None:
    attn = torch.softmax(torch.randn(2, 4, 5), dim=-1)
    target = torch.ones(2, 4, 5) / 5
    assert grounding_kl(attn, target).ndim == 0
