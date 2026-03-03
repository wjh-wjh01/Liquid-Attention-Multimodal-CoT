from __future__ import annotations

import pathlib
import random
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.common.config import save_yaml
from src.common.io_utils import read_jsonl, write_json, write_jsonl
from src.common.logging_utils import log_line
from src.data.dataset import DatasetConfig, JsonlReasoningDataset
from src.data.tokenizer import SimpleTokenizer
from src.models.multimodal_cot import MultimodalCoTModel


@dataclass
class RunArtifacts:
    run_dir: pathlib.Path
    checkpoint_path: pathlib.Path
    tokenizer_path: pathlib.Path
    train_metrics_path: pathlib.Path
    eval_metrics_path: pathlib.Path
    robustness_path: pathlib.Path
    efficiency_path: pathlib.Path


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda") 
    if torch.backends.mps.is_available():
        return torch.device("mps")   
    return torch.device("cpu")    


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Build tokenizer from training data
def build_tokenizer(train_jsonl: pathlib.Path, max_vocab_size: int) -> SimpleTokenizer:
    print("Building vocabulary from training data...")
    from datasets import load_dataset
    texts = []
    try:
        # Try loading full training set from parquet
        parquet_path = "/root/autodl-tmp/project/data/raw/scienceqa_full/data/train-full.parquet"
        ds = load_dataset("parquet", data_files={"train": parquet_path})["train"]
        for row in ds:
            texts.append(str(row.get("question", "")))
            choices = row.get("choices", [])
            if choices is not None:
                for c in choices:
                    texts.append(str(c))
    except Exception as e:
        print(f"Parquet load failed, fallback to jsonl: {e}")
        from src.common.io_utils import read_jsonl
        rows = read_jsonl(train_jsonl)
        for r in rows:
            texts.append(str(r.get("question", "")))
            for c in r.get("choices", []):
                texts.append(str(c))
                
    tok = SimpleTokenizer()
    tok.fit(texts, max_vocab_size=max_vocab_size)
    print("Vocabulary built.")
    return tok


def _move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _compute_classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    valid = labels >= 0
    if valid.sum().item() == 0:
        return {"accuracy": 0.0, "count": 0.0}

    pred = torch.argmax(logits[valid], dim=-1)
    acc = (pred == labels[valid]).float().mean().item()
    return {"accuracy": float(acc), "count": float(valid.sum().item())}


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    flags: Dict[str, bool],
    split_name: str,
) -> Dict[str, Any]:
    model.eval()
    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0.0
    total_count = 0.0
    total_steps = 0.0
    total_entropy = 0.0
    total_stop = 0.0
    pred_rows: List[Dict[str, Any]] = []

    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            out = model(batch, flags=flags)
            labels = batch["answer_idx"]

            valid = labels >= 0
            if valid.sum().item() > 0:
                loss = ce(out["logits"][valid], labels[valid])
                total_loss += loss.item() * valid.sum().item()

                preds = torch.argmax(out["logits"][valid], dim=-1)
                total_correct += (preds == labels[valid]).float().sum().item()
                total_count += valid.sum().item()

            total_steps += out["steps_used"].float().sum().item()
            total_entropy += out["attn_entropy"].float().sum().item()
            total_stop += out["stop_prob"].float().sum().item()

            all_preds = torch.argmax(out["logits"], dim=-1).detach().cpu().tolist()
            all_labels = labels.detach().cpu().tolist()
            all_stop = out["stop_prob"].detach().cpu().tolist()
            all_steps = out["steps_used"].detach().cpu().tolist()
            difficulties = batch.get("difficulty", [None] * len(all_preds))

            for sid, p, y, sp, st, diff in zip(batch["id"], all_preds, all_labels, all_stop, all_steps, difficulties):
                pred_rows.append(
                    {
                        "id": sid,
                        "split": split_name,
                        "pred": int(p),
                        "label": int(y),
                        "correct": int(p == y and y >= 0),
                        "stop_prob": float(sp),
                        "steps_used": int(st),
                        "difficulty": diff,
                    }
                )

    elapsed = time.perf_counter() - start
    denom = max(total_count, 1.0)
    sample_count = max(len(pred_rows), 1)

    return {
        "split": split_name,
        "loss": float(total_loss / denom),
        "accuracy": float(total_correct / denom),
        "count": int(total_count),
        "avg_steps": float(total_steps / sample_count),
        "avg_attn_entropy": float(total_entropy / sample_count),
        "avg_stop_prob": float(total_stop / sample_count),
        "eval_time_sec": float(elapsed),
        "predictions": pred_rows,
    }


def _difficulty_accuracy(pred_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    easy_keys = {"easy", "simple", "low"}
    hard_keys = {"hard", "difficult", "difficulty", "high", "complex"}

    easy = []
    hard = []
    unknown = []

    for r in pred_rows:
        if int(r.get("label", -1)) < 0:
            continue
        diff = r.get("difficulty")
        tag = str(diff).strip().lower() if diff is not None else ""
        if tag in easy_keys:
            easy.append(r)
        elif tag in hard_keys:
            hard.append(r)
        else:
            unknown.append(r)

    def acc(rows: List[Dict[str, Any]]) -> float:
        if not rows:
            return 0.0
        return float(sum(int(x["correct"]) for x in rows) / len(rows))

    if not easy and unknown:
        midpoint = len(unknown) // 2
        easy = unknown[:midpoint]
        hard = unknown[midpoint:]

    return {
        "simple_acc": acc(easy),
        "difficulty_acc": acc(hard),
    }


def _build_loader(
    split_path: pathlib.Path,
    tokenizer: SimpleTokenizer,
    model_cfg: Dict[str, Any],
    batch_size: int,
    include_unlabeled: bool,
    noise_prob: float,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    ds_cfg = DatasetConfig(
        max_question_len=int(model_cfg["max_question_len"]),
        max_choice_len=int(model_cfg["max_choice_len"]),
        image_dim=int(model_cfg["image_dim"]),
        noise_prob=float(noise_prob),
    )
    ds = JsonlReasoningDataset(
        jsonl_path=str(split_path),
        tokenizer=tokenizer,
        cfg=ds_cfg,
        include_unlabeled=include_unlabeled,
        seed=seed,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=ds.collate_fn)


def run_single_experiment(
    run_cfg: Dict[str, Any],
    experiment_name: str,
    dataset_name: str,
    flags_override: Optional[Dict[str, bool]] = None,
    model_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    seed = int(run_cfg.get("seed", 3407))
    seed_everything(seed)

    dataset_cfg = dict(run_cfg["dataset"])
    model_cfg = dict(run_cfg["model"])
    train_cfg = dict(run_cfg["train"])
    eval_cfg = dict(run_cfg["eval"])
    flags = dict(run_cfg["ablation_flags"])

    if flags_override:
        flags.update(flags_override)
    if model_override:
        model_cfg.update(model_override)

    dataset_cfg["name"] = dataset_name

    processed_root = pathlib.Path(dataset_cfg.get("processed_root", "data/processed"))
    ds_dir = processed_root / dataset_name

    train_jsonl = ds_dir / "train.jsonl"
    val_jsonl = ds_dir / "val.jsonl"
    test_jsonl = ds_dir / "test.jsonl"
    if not train_jsonl.exists() or not test_jsonl.exists():
        raise FileNotFoundError(
            f"Processed dataset files missing for {dataset_name}. "
            f"Expected: {train_jsonl} and {test_jsonl}. Run scripts/prepare_data.py first."
        )

    output_root = pathlib.Path(run_cfg.get("output_dir", "outputs/runs"))
    run_dir = output_root / dataset_name / experiment_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    artifacts = RunArtifacts(
        run_dir=run_dir,
        checkpoint_path=run_dir / "checkpoint.pt",
        tokenizer_path=run_dir / "tokenizer.json",
        train_metrics_path=run_dir / "train_metrics.json",
        eval_metrics_path=run_dir / "eval_metrics.json",
        robustness_path=run_dir / "robustness_metrics.json",
        efficiency_path=run_dir / "efficiency.json",
    )

    save_yaml(
        run_dir / "config_snapshot.yaml",
        {
            **run_cfg,
            "dataset": dataset_cfg,
            "model": model_cfg,
            "ablation_flags": flags,
            "experiment_name": experiment_name,
        },
    )

    tokenizer = build_tokenizer(train_jsonl, max_vocab_size=int(model_cfg.get("vocab_size", 20000)))
    tokenizer.save(artifacts.tokenizer_path)

    train_loader = _build_loader(
        split_path=train_jsonl,
        tokenizer=tokenizer,
        model_cfg=model_cfg,
        batch_size=int(train_cfg.get("batch_size", 16)),
        include_unlabeled=False,
        noise_prob=0.0,
        seed=seed,
        shuffle=True,
    )
    val_loader = _build_loader(
        split_path=val_jsonl if val_jsonl.exists() else test_jsonl,
        tokenizer=tokenizer,
        model_cfg=model_cfg,
        batch_size=int(eval_cfg.get("batch_size", 32)),
        include_unlabeled=False,
        noise_prob=0.0,
        seed=seed,
        shuffle=False,
    )
    test_loader = _build_loader(
        split_path=test_jsonl,
        tokenizer=tokenizer,
        model_cfg=model_cfg,
        batch_size=int(eval_cfg.get("batch_size", 32)),
        include_unlabeled=False,
        noise_prob=0.0,
        seed=seed,
        shuffle=False,
    )

    device = select_device()
    log_line(f"Running {dataset_name}/{experiment_name} on device={device}")

    model = MultimodalCoTModel(model_cfg=model_cfg, default_flags=flags).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    ce = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    epoch_logs: List[Dict[str, Any]] = []

    tracemalloc.start()
    train_start = time.perf_counter()

    for epoch in range(int(train_cfg.get("epochs", 3))):
        model.train()
        total_loss = 0.0
        total_correct = 0.0
        total_count = 0.0
        total_seen = 0

        for step, batch in enumerate(train_loader, start=1):
            batch = _move_batch(batch, device)
            out = model(batch, flags=flags)
            labels = batch["answer_idx"]
            valid = labels >= 0
            if valid.sum().item() == 0:
                continue

            ce_loss = ce(out["logits"][valid], labels[valid])
            stop_reg = (1.0 - out["stop_prob"].mean()) if flags.get("use_self_validation", True) else 0.0
            loss = ce_loss + 0.05 * stop_reg

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(train_cfg.get("grad_clip", 1.0)))
            optimizer.step()

            with torch.no_grad():
                preds = torch.argmax(out["logits"][valid], dim=-1)
                total_correct += (preds == labels[valid]).float().sum().item()
                total_count += valid.sum().item()

            total_loss += float(loss.item()) * valid.sum().item()
            total_seen += int(valid.sum().item())

            if step % int(train_cfg.get("log_every", 20)) == 0:
                log_line(
                    f"[{dataset_name}/{experiment_name}] epoch={epoch+1} step={step} "
                    f"loss={total_loss/max(total_count,1):.4f} acc={total_correct/max(total_count,1):.4f}"
                )

        train_epoch_loss = float(total_loss / max(total_count, 1.0))
        train_epoch_acc = float(total_correct / max(total_count, 1.0))

        val_metrics = evaluate_model(model, val_loader, device, flags, split_name="val")
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": train_epoch_loss,
            "train_accuracy": train_epoch_acc,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        epoch_logs.append(epoch_log)

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = float(val_metrics["accuracy"])
            torch.save({"model_state": model.state_dict(), "epoch": epoch + 1}, artifacts.checkpoint_path)

    train_time = time.perf_counter() - train_start
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if artifacts.checkpoint_path.exists():
        ckpt = torch.load(artifacts.checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    test_metrics = evaluate_model(model, test_loader, device, flags, split_name="test")

    robustness = {}
    diff_stats = _difficulty_accuracy(test_metrics["predictions"])
    robustness.update(diff_stats)

    for noise in eval_cfg.get("noise_levels", [0.1, 0.2]):
        noise_loader = _build_loader(
            split_path=test_jsonl,
            tokenizer=tokenizer,
            model_cfg=model_cfg,
            batch_size=int(eval_cfg.get("batch_size", 32)),
            include_unlabeled=False,
            noise_prob=float(noise),
            seed=seed,
            shuffle=False,
        )
        noise_metrics = evaluate_model(model, noise_loader, device, flags, split_name=f"test_noise_{noise}")
        robustness[f"noise_{int(noise*100)}_acc"] = float(noise_metrics["accuracy"])

    train_summary = {
        "dataset": dataset_name,
        "experiment": experiment_name,
        "seed": seed,
        "epochs": int(train_cfg.get("epochs", 3)),
        "best_val_accuracy": best_val_acc,
        "epoch_logs": epoch_logs,
    }

    eval_summary = {
        "dataset": dataset_name,
        "experiment": experiment_name,
        "seed": seed,
        "test": {k: v for k, v in test_metrics.items() if k != "predictions"},
    }

    efficiency = {
        "dataset": dataset_name,
        "experiment": experiment_name,
        "seed": seed,
        "train_time_sec": float(train_time),
        "eval_time_sec": float(test_metrics["eval_time_sec"]),
        "peak_memory_mb": float(peak_mem / (1024 * 1024)),
        "avg_steps": float(test_metrics["avg_steps"]),
        "accuracy": float(test_metrics["accuracy"]),
    }

    write_json(artifacts.train_metrics_path, train_summary)
    write_json(artifacts.eval_metrics_path, eval_summary)
    write_json(artifacts.robustness_path, robustness)
    write_json(artifacts.efficiency_path, efficiency)
    write_jsonl(run_dir / "predictions_test.jsonl", test_metrics["predictions"])

    return {
        "dataset": dataset_name,
        "experiment": experiment_name,
        "seed": seed,
        "run_dir": str(run_dir),
        "best_val_accuracy": best_val_acc,
        "test_accuracy": float(test_metrics["accuracy"]),
        "avg_steps": float(test_metrics["avg_steps"]),
    }
