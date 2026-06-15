from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import torch

State = Tuple[torch.Tensor, torch.Tensor]
VectorField = Callable[[float, State], State]


@dataclass(frozen=True)
class SolverConfig:
    name: str = "rk4"
    horizon: float = 1.0
    steps: int = 12


def _add_state(a: State, b: State, scale: float = 1.0) -> State:
    return a[0] + scale * b[0], a[1] + scale * b[1]


def euler(field: VectorField, state: State, cfg: SolverConfig) -> State:
    dt = cfg.horizon / cfg.steps
    t = 0.0
    h, s = state
    for _ in range(cfg.steps):
        dh, ds = field(t, (h, s))
        h, s = h + dt * dh, s + dt * ds
        t += dt
    return h, s


def midpoint(field: VectorField, state: State, cfg: SolverConfig) -> State:
    dt = cfg.horizon / cfg.steps
    t = 0.0
    h, s = state
    for _ in range(cfg.steps):
        k1 = field(t, (h, s))
        mid = _add_state((h, s), k1, 0.5 * dt)
        k2 = field(t + 0.5 * dt, mid)
        h, s = _add_state((h, s), k2, dt)
        t += dt
    return h, s


def rk4(field: VectorField, state: State, cfg: SolverConfig) -> State:
    dt = cfg.horizon / cfg.steps
    t = 0.0
    h, s = state
    for _ in range(cfg.steps):
        k1 = field(t, (h, s))
        k2 = field(t + 0.5 * dt, _add_state((h, s), k1, 0.5 * dt))
        k3 = field(t + 0.5 * dt, _add_state((h, s), k2, 0.5 * dt))
        k4 = field(t + dt, _add_state((h, s), k3, dt))
        h = h + (dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        s = s + (dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        t += dt
    return h, s


def solve(field: VectorField, state: State, cfg: SolverConfig) -> State:
    if cfg.name == "euler":
        return euler(field, state, cfg)
    if cfg.name == "midpoint":
        return midpoint(field, state, cfg)
    if cfg.name == "rk4":
        return rk4(field, state, cfg)
    raise ValueError(f"Unsupported fixed-step solver: {cfg.name}")
