from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ct_mcot.data import JsonlMemoryDataset, collate_memory
from ct_mcot.model import CTMCoTConfig, CTMCoTModel
from ct_mcot.training.callbacks import MetricsLogger
from ct_mcot.training.optim import build_optimizer, build_scheduler
from ct_mcot.training.trainer import Trainer
from ct_mcot.utils.config import load_experiment_config, save_yaml
from ct_mcot.utils.seed import set_seed


def run_experiment(config_paths: list[str | Path], overrides: list[str] | None = None) -> None:
    from ct_mcot.utils.config import apply_overrides

    cfg = load_experiment_config(config_paths)
    cfg = apply_overrides(cfg, overrides or [])
    seed = int(cfg.get("train", {}).get("seed", 13))
    set_seed(seed, deterministic=bool(cfg.get("train", {}).get("deterministic", False)))
    output_dir = Path(cfg.get("train", {}).get("output_dir", "outputs/experiment"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, output_dir / "resolved_config.yaml")

    dataset = JsonlMemoryDataset(cfg["data"]["train_path"])
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"].get("batch_size", 32)),
        shuffle=True,
        collate_fn=collate_memory,
        num_workers=int(cfg["train"].get("num_workers", 0)),
    )
    model = CTMCoTModel(CTMCoTConfig(**cfg["model"]))
    optimizer = build_optimizer(model, cfg.get("optimizer", cfg.get("train", {})))
    total_steps = len(loader) * int(cfg["train"].get("epochs", 5))
    scheduler = build_scheduler(optimizer, cfg.get("scheduler", {"name": "none"}), total_steps)
    device = torch.device(cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=nn.CrossEntropyLoss(),
        device=device,
        output_dir=output_dir,
        callbacks=[MetricsLogger(output_dir / "train_log.jsonl")],
        clip_grad_norm=float(cfg["train"].get("clip_grad_norm", 1.0)),
        amp=bool(cfg["train"].get("amp", False)),
    )
    trainer.fit(loader, int(cfg["train"].get("epochs", 5)), cfg)
