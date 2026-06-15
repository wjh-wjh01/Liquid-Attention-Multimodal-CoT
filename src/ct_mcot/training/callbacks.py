from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ct_mcot.utils.logging import JsonlLogger


@dataclass
class CallbackState:
    epoch: int
    step: int
    metrics: dict[str, float]


class Callback(Protocol):
    def on_epoch_end(self, state: CallbackState) -> None: ...

    def on_step_end(self, state: CallbackState) -> None: ...


class MetricsLogger:
    def __init__(self, path: str | Path):
        self.logger = JsonlLogger(path)
        self.start = time.time()

    def on_step_end(self, state: CallbackState) -> None:
        row = {"event": "step", "elapsed_sec": time.time() - self.start, **state.__dict__}
        self.logger.write(row)

    def on_epoch_end(self, state: CallbackState) -> None:
        row = {"event": "epoch", "elapsed_sec": time.time() - self.start, **state.__dict__}
        self.logger.write(row)


class EarlyStopping:
    def __init__(self, metric: str = "eval_accuracy", patience: int = 3, maximize: bool = True):
        self.metric = metric
        self.patience = patience
        self.maximize = maximize
        self.best: float | None = None
        self.bad_epochs = 0
        self.should_stop = False

    def on_step_end(self, state: CallbackState) -> None:
        return None

    def on_epoch_end(self, state: CallbackState) -> None:
        if self.metric not in state.metrics:
            return
        value = state.metrics[self.metric]
        improved = self.best is None or (value > self.best if self.maximize else value < self.best)
        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            self.should_stop = self.bad_epochs >= self.patience
