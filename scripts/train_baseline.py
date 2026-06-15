#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from ct_mcot.baselines import build_baseline
from ct_mcot.data import JsonlMemoryDataset, collate_memory


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a controlled baseline on JSONL memory tokens.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline", required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device(cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    dataset = JsonlMemoryDataset(cfg["data"]["train_path"])
    loader = DataLoader(dataset, batch_size=cfg["train"].get("batch_size", 32), shuffle=True, collate_fn=collate_memory)
    model = build_baseline(args.baseline, cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"].get("lr", 3e-4)))
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(int(cfg["train"].get("epochs", 5))):
        total = 0
        correct = 0
        for batch in loader:
            out = model(batch["memory"].to(device), batch["mask"].to(device))
            label = batch["label"].to(device)
            loss = loss_fn(out["logits"], label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += label.numel()
            correct += (out["logits"].argmax(-1) == label).sum().item()
        print({"epoch": epoch + 1, "accuracy": correct / max(total, 1)})


if __name__ == "__main__":
    main()
