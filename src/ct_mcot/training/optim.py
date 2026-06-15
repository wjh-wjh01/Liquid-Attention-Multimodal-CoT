from __future__ import annotations

import math

import torch


def build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    name = cfg.get("name", "adamw").lower()
    lr = float(cfg.get("lr", 3e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-2))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict, total_steps: int):
    name = cfg.get("name", "cosine").lower()
    warmup = int(cfg.get("warmup_steps", 0))
    if name == "none":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    if name == "linear":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min(1.0, (step + 1) / max(warmup, 1))
            if step < warmup
            else max(0.0, (total_steps - step) / max(total_steps - warmup, 1)),
        )
    if name == "cosine":
        def factor(step: int) -> float:
            if step < warmup:
                return min(1.0, (step + 1) / max(warmup, 1))
            progress = (step - warmup) / max(total_steps - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    raise ValueError(f"Unsupported scheduler: {name}")
