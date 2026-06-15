from __future__ import annotations

import math

import numpy as np


def paired_bootstrap_difference(
    a_correct: list[int] | np.ndarray,
    b_correct: list[int] | np.ndarray,
    samples: int = 10000,
    seed: int = 13,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    a = np.asarray(a_correct).astype(float)
    b = np.asarray(b_correct).astype(float)
    diffs = []
    for _ in range(samples):
        idx = rng.integers(0, len(a), len(a))
        diffs.append(float(a[idx].mean() - b[idx].mean()))
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci95_low": float(np.percentile(diffs, 2.5)),
        "ci95_high": float(np.percentile(diffs, 97.5)),
        "p_two_sided": float(2 * min(np.mean(np.asarray(diffs) <= 0), np.mean(np.asarray(diffs) >= 0))),
    }


def mcnemar_table(
    gold: list[int] | np.ndarray,
    pred_a: list[int] | np.ndarray,
    pred_b: list[int] | np.ndarray,
) -> dict[str, float]:
    gold = np.asarray(gold)
    a = np.asarray(pred_a) == gold
    b = np.asarray(pred_b) == gold
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    statistic = (abs(n01 - n10) - 1) ** 2 / max(n01 + n10, 1)
    p_approx = math.exp(-0.5 * statistic)
    return {"n01": n01, "n10": n10, "chi2_cc": statistic, "p_approx": p_approx}
