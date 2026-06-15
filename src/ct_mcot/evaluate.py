from __future__ import annotations

import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from .data import JsonlMemoryDataset, collate_memory
from .train import build_model, load_config


@torch.no_grad()
def evaluate_from_config(config_path: str | Path, checkpoint_path: str | Path | None = None) -> dict:
    cfg = load_config(config_path)
    device = torch.device(cfg["eval"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(checkpoint_path or cfg["eval"]["checkpoint_path"])
    output_dir = Path(cfg["eval"].get("output_dir", "outputs/synthetic"))
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = JsonlMemoryDataset(cfg["data"]["eval_path"])
    loader = DataLoader(
        dataset,
        batch_size=cfg["eval"].get("batch_size", 64),
        shuffle=False,
        collate_fn=collate_memory,
    )
    y_true: list[int] = []
    y_pred: list[int] = []
    pred_rows = []
    for batch in loader:
        output = model(batch["memory"].to(device), batch["mask"].to(device), return_diagnostics=True)
        probs = torch.softmax(output["logits"], dim=-1).cpu()
        pred = probs.argmax(dim=-1)
        labels = batch["label"]
        y_true.extend(labels.tolist())
        y_pred.extend(pred.tolist())
        for row_id, label, item_pred, prob in zip(batch["id"], labels, pred, probs):
            pred_rows.append(
                {
                    "id": row_id,
                    "label": int(label),
                    "prediction": int(item_pred),
                    "probabilities": [float(x) for x in prob],
                }
            )

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "num_examples": len(y_true),
    }
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in pred_rows:
            f.write(json.dumps(row) + "\n")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    return metrics
