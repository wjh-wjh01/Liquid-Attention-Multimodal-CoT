from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def load_predictions(path: str | Path) -> tuple[list[int], list[int]]:
    y_true, y_pred = [], []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            y_true.append(int(row["label"]))
            y_pred.append(int(row["prediction"]))
    return y_true, y_pred


def bootstrap_accuracy(y_true: list[int], y_pred: list[int], samples: int = 1000, seed: int = 13) -> dict:
    rng = np.random.default_rng(seed)
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    scores = []
    for _ in range(samples):
        idx = rng.integers(0, len(y_true_arr), len(y_true_arr))
        scores.append(float(accuracy_score(y_true_arr[idx], y_pred_arr[idx])))
    return {
        "mean": float(np.mean(scores)),
        "ci95_low": float(np.percentile(scores, 2.5)),
        "ci95_high": float(np.percentile(scores, 97.5)),
    }


def summarize_predictions(path: str | Path) -> dict:
    y_true, y_pred = load_predictions(path)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "bootstrap_accuracy": bootstrap_accuracy(y_true, y_pred),
    }
