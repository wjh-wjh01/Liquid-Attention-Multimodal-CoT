from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import JsonlMemoryDataset, collate_memory
from .model import CTMCoTConfig, CTMCoTModel


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> CTMCoTModel:
    return CTMCoTModel(CTMCoTConfig(**cfg["model"]))


def train_from_config(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    device = torch.device(cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg["train"].get("seed", 13))
    output_dir = Path(cfg["train"].get("output_dir", "outputs/synthetic"))
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = JsonlMemoryDataset(cfg["data"]["train_path"])
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"].get("batch_size", 32),
        shuffle=True,
        collate_fn=collate_memory,
    )
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"].get("lr", 3e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-2)),
    )
    loss_fn = nn.CrossEntropyLoss()
    epochs = int(cfg["train"].get("epochs", 5))
    clip_norm = float(cfg["train"].get("clip_grad_norm", 1.0))

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}")
        for batch in progress:
            memory = batch["memory"].to(device)
            mask = batch["mask"].to(device)
            label = batch["label"].to(device)
            logits = model(memory, mask)["logits"]
            loss = loss_fn(logits, label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            total_loss += loss.item() * label.numel()
            pred = logits.argmax(dim=-1)
            correct += (pred == label).sum().item()
            total += label.numel()
            progress.set_postfix(loss=total_loss / total, acc=correct / total)

    checkpoint = {
        "model": model.state_dict(),
        "config": cfg,
    }
    torch.save(checkpoint, output_dir / "ct_mcot.pt")
    print(f"saved checkpoint: {output_dir / 'ct_mcot.pt'}")
