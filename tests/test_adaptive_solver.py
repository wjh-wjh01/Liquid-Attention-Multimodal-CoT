from __future__ import annotations

import torch

from ct_mcot.adaptive_solvers import AdaptiveSolverConfig, dopri5


def test_dopri5_reaches_simple_decay() -> None:
    def field(_t, state):
        h, s = state
        return -h, -s

    h0 = torch.ones(2, 3)
    s0 = torch.ones(2, 4)
    result = dopri5(field, (h0, s0), AdaptiveSolverConfig(horizon=0.1, max_nfe=32))
    assert result.nfe > 0
    assert result.state[0].mean() < h0.mean()
