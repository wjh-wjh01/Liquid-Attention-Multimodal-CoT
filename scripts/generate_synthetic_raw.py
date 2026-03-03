#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import random


def make_rows(dataset: str, split: str, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        a = rng.randint(1, 30)
        b = rng.randint(1, 30)
        c0 = a + b - 1
        c1 = a + b
        c2 = a + b + 1
        c3 = a + b + 2
        choices = [str(c0), str(c1), str(c2), str(c3)]

        if dataset == "scienceqa":
            q = f"In science class, what is {a}+{b}?"
        elif dataset == "mmlu_pro":
            q = f"Professional reasoning: compute {a}+{b}."
        else:
            q = f"Multimodal chain-of-thought question: {a}+{b}=?"

        rows.append(
            {
                "question": q,
                "choices": choices,
                "answer_idx": 1,
                "split": split,
                "difficulty": "simple" if i % 2 == 0 else "difficulty",
                "image_path": None,
                "metadata": {"synthetic": True},
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic local raw datasets for smoke testing")
    parser.add_argument("--raw-root", type=str, default="data/raw")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--train", type=int, default=120)
    parser.add_argument("--val", type=int, default=30)
    parser.add_argument("--test", type=int, default=40)
    args = parser.parse_args()

    root = pathlib.Path(args.raw_root)
    for dataset in ["scienceqa", "mmlu_pro", "cmmcot"]:
        ds_dir = root / dataset
        ds_dir.mkdir(parents=True, exist_ok=True)
        for split, n in [("train", args.train), ("val", args.val), ("test", args.test)]:
            rows = make_rows(dataset, split, n, seed=args.seed + hash((dataset, split)) % 1000)
            with (ds_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print({"raw_root": str(root), "datasets": ["scienceqa", "mmlu_pro", "cmmcot"]})


if __name__ == "__main__":
    main()
