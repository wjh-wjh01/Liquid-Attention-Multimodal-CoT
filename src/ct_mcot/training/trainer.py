from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ct_mcot.utils.checkpointing import save_checkpoint

from .callbacks import Callback, CallbackState


@dataclass
class TrainerState:
    epoch: int = 0
    step: int = 0
    best_metric: float | None = None


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        output_dir: str | Path,
        scheduler=None,
        callbacks: list[Callback] | None = None,
        clip_grad_norm: float = 1.0,
        amp: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.output_dir = Path(output_dir)
        self.callbacks = callbacks or []
        self.clip_grad_norm = clip_grad_norm
        self.amp = amp
        self.state = TrainerState()
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp)

    def fit(self, loader: DataLoader, epochs: int, cfg: dict) -> None:
        self.model.to(self.device)
        for epoch in range(epochs):
            self.state.epoch = epoch
            metrics = self._train_epoch(loader, epoch, epochs)
            callback_state = CallbackState(epoch=epoch, step=self.state.step, metrics=metrics)
            for callback in self.callbacks:
                callback.on_epoch_end(callback_state)
            save_checkpoint(
                self.output_dir / f"checkpoint_epoch_{epoch + 1}.pt",
                self.model,
                self.optimizer,
                cfg,
                self.state.step,
                metrics,
            )

    def _train_epoch(self, loader: DataLoader, epoch: int, epochs: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        count = 0
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}")
        for batch in progress:
            memory = batch["memory"].to(self.device)
            mask = batch["mask"].to(self.device)
            label = batch["label"].to(self.device)
            with torch.cuda.amp.autocast(enabled=self.amp):
                output = self.model(memory, mask)
                loss = self.loss_fn(output["logits"], label)
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()

            self.state.step += 1
            batch_size = label.numel()
            total_loss += loss.item() * batch_size
            correct += (output["logits"].argmax(dim=-1) == label).sum().item()
            count += batch_size
            metrics = {"train_loss": total_loss / count, "train_accuracy": correct / count}
            progress.set_postfix(metrics)
            callback_state = CallbackState(epoch=epoch, step=self.state.step, metrics=metrics)
            for callback in self.callbacks:
                callback.on_step_end(callback_state)
        return {"train_loss": total_loss / count, "train_accuracy": correct / count}
