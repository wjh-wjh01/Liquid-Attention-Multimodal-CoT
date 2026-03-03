#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from src.common.config import load_yaml
from src.common.io_utils import write_json, write_jsonl
from src.data.dataset import DatasetConfig, JsonlReasoningDataset
from src.data.tokenizer import SimpleTokenizer
from src.models.multimodal_cot import MultimodalCoTModel
from src.training.pipeline import evaluate_model, select_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an existing checkpoint")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--noise", type=float, default=0.0)
    args = parser.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    cfg = load_yaml(run_dir / "config_snapshot.yaml")
    model_cfg = cfg["model"]
    flags = cfg["ablation_flags"]
    dataset_name = cfg["dataset"]["name"]

    tokenizer = SimpleTokenizer.load(run_dir / "tokenizer.json")
    split_path = pathlib.Path(cfg["dataset"].get("processed_root", "data/processed")) / dataset_name / f"{args.split}.jsonl"

    ds_cfg = DatasetConfig(
        max_question_len=int(model_cfg["max_question_len"]),
        max_choice_len=int(model_cfg["max_choice_len"]),
        image_dim=int(model_cfg["image_dim"]),
        noise_prob=float(args.noise),
    )
    ds = JsonlReasoningDataset(str(split_path), tokenizer, ds_cfg, include_unlabeled=False, seed=int(cfg.get("seed", 3407)))
    loader = DataLoader(ds, batch_size=int(cfg["eval"].get("batch_size", 32)), shuffle=False, collate_fn=ds.collate_fn)

    device = select_device()
    model = MultimodalCoTModel(model_cfg=model_cfg, default_flags=flags).to(device)
    ckpt = torch.load(run_dir / "checkpoint.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    metrics = evaluate_model(model, loader, device, flags, split_name=args.split)
    out_json = run_dir / f"eval_metrics_{args.split}_noise_{args.noise:.2f}.json"
    out_pred = run_dir / f"predictions_{args.split}_noise_{args.noise:.2f}.jsonl"
    write_json(out_json, {k: v for k, v in metrics.items() if k != "predictions"})
    write_jsonl(out_pred, metrics["predictions"])
    print({"saved_metrics": str(out_json), "accuracy": metrics["accuracy"]})


if __name__ == "__main__":
    main()
