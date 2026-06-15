from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import torch

State = Tuple[torch.Tensor, torch.Tensor]
VectorField = Callable[[float, State], State]


@dataclass
class AdaptiveSolverConfig:
    horizon: float = 1.0
    rtol: float = 1e-3
    atol: float = 1e-4
    max_nfe: int = 32
    min_step: float = 1e-4
    max_step: float = 0.25
    safety: float = 0.9


@dataclass
class AdaptiveResult:
    state: State
    nfe: int
    accepted_steps: int
    rejected_steps: int
    hit_cap: bool


def dopri5(field: VectorField, state: State, cfg: AdaptiveSolverConfig) -> AdaptiveResult:
    """Dormand-Prince 5(4) solver for two-tensor CT-MCoT states."""
    t = 0.0
    dt = min(cfg.max_step, cfg.horizon)
    y = state
    nfe = 0
    accepted = 0
    rejected = 0
    while t < cfg.horizon and nfe < cfg.max_nfe:
        dt = min(dt, cfg.horizon - t)
        y5, y4, evals = _dopri_step(field, t, y, dt)
        nfe += evals
        err = _error_ratio(y5, y4, cfg.atol, cfg.rtol)
        if err <= 1.0 or dt <= cfg.min_step:
            t += dt
            y = y5
            accepted += 1
            dt = _next_step(dt, err, cfg.safety, cfg.max_step, cfg.min_step)
        else:
            rejected += 1
            dt = _next_step(dt, err, cfg.safety, cfg.max_step, cfg.min_step)
    return AdaptiveResult(y, nfe, accepted, rejected, hit_cap=t < cfg.horizon)


def _dopri_step(field: VectorField, t: float, y: State, dt: float) -> tuple[State, State, int]:
    k1 = field(t, y)
    k2 = field(t + dt * 1 / 5, _combine(y, [(dt * 1 / 5, k1)]))
    k3 = field(t + dt * 3 / 10, _combine(y, [(dt * 3 / 40, k1), (dt * 9 / 40, k2)]))
    k4 = field(t + dt * 4 / 5, _combine(y, [(dt * 44 / 45, k1), (dt * -56 / 15, k2), (dt * 32 / 9, k3)]))
    k5 = field(
        t + dt * 8 / 9,
        _combine(y, [(dt * 19372 / 6561, k1), (dt * -25360 / 2187, k2), (dt * 64448 / 6561, k3), (dt * -212 / 729, k4)]),
    )
    k6 = field(
        t + dt,
        _combine(y, [(dt * 9017 / 3168, k1), (dt * -355 / 33, k2), (dt * 46732 / 5247, k3), (dt * 49 / 176, k4), (dt * -5103 / 18656, k5)]),
    )
    k7 = field(
        t + dt,
        _combine(y, [(dt * 35 / 384, k1), (dt * 500 / 1113, k3), (dt * 125 / 192, k4), (dt * -2187 / 6784, k5), (dt * 11 / 84, k6)]),
    )
    y5 = _combine(y, [(dt * 35 / 384, k1), (dt * 500 / 1113, k3), (dt * 125 / 192, k4), (dt * -2187 / 6784, k5), (dt * 11 / 84, k6)])
    y4 = _combine(
        y,
        [
            (dt * 5179 / 57600, k1),
            (dt * 7571 / 16695, k3),
            (dt * 393 / 640, k4),
            (dt * -92097 / 339200, k5),
            (dt * 187 / 2100, k6),
            (dt * 1 / 40, k7),
        ],
    )
    return y5, y4, 7


def _combine(base: State, terms: list[tuple[float, State]]) -> State:
    h, s = base
    for scale, state in terms:
        h = h + scale * state[0]
        s = s + scale * state[1]
    return h, s


def _error_ratio(a: State, b: State, atol: float, rtol: float) -> float:
    ratios = []
    for aa, bb in zip(a, b):
        scale = atol + rtol * torch.maximum(aa.abs(), bb.abs())
        ratios.append(((aa - bb) / scale).pow(2).mean().sqrt())
    return float(torch.stack(ratios).max().detach().cpu())


def _next_step(dt: float, err: float, safety: float, max_step: float, min_step: float) -> float:
    if err == 0:
        factor = 2.0
    else:
        factor = min(2.0, max(0.2, safety * err ** (-0.2)))
    return min(max_step, max(min_step, dt * factor))
