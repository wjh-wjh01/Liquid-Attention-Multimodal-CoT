from __future__ import annotations

import torch

from ct_mcot.baselines import build_baseline


def test_direct_baseline_shape() -> None:
    model = build_baseline("no_cot", {"input_dim": 8, "hidden_dim": 16, "num_classes": 3})
    out = model(torch.randn(2, 5, 8), torch.ones(2, 5, dtype=torch.bool))
    assert out["logits"].shape == (2, 3)
