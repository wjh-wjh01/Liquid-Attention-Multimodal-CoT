from __future__ import annotations

import torch

from ct_mcot import CTMCoTConfig, CTMCoTModel


def test_model_forward_shapes() -> None:
    cfg = CTMCoTConfig(input_dim=16, hidden_dim=32, thought_slots=4, steps=2)
    model = CTMCoTModel(cfg)
    memory = torch.randn(3, 10, 16)
    mask = torch.ones(3, 10, dtype=torch.bool)
    output = model(memory, mask, return_diagnostics=True)
    assert output["logits"].shape == (3, 2)
    assert output["thought"].shape == (3, 4, 32)
    assert output["attention"].shape == (3, 4, 10)
