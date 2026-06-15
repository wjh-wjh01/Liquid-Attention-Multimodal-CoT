#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from ct_mcot.evaluation.significance import mcnemar_table, paired_bootstrap_difference


def read(path: str) -> tuple[list[int], list[int]]:
    gold, pred = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            gold.append(int(row["label"]))
            pred.append(int(row["prediction"]))
    return gold, pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired significance comparison for two prediction files.")
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--samples", type=int, default=10000)
    args = parser.parse_args()
    gold_a, pred_a = read(args.a)
    gold_b, pred_b = read(args.b)
    if gold_a != gold_b:
        raise ValueError("Prediction files must use the same examples in the same order.")
    a_correct = [int(p == g) for p, g in zip(pred_a, gold_a)]
    b_correct = [int(p == g) for p, g in zip(pred_b, gold_a)]
    result = {
        "bootstrap": paired_bootstrap_difference(a_correct, b_correct, samples=args.samples),
        "mcnemar": mcnemar_table(gold_a, pred_a, pred_b),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
