#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_yaml
from src.common.io_utils import read_json
from src.training.pipeline import run_single_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate one experiment")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--flags-json", type=str, default="")
    parser.add_argument("--model-json", type=str, default="")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    flags_override = read_json(args.flags_json) if args.flags_json else None
    model_override = read_json(args.model_json) if args.model_json else None

    result = run_single_experiment(
        run_cfg=cfg,
        experiment_name=args.experiment,
        dataset_name=args.dataset,
        flags_override=flags_override,
        model_override=model_override,
    )
    print(result)


if __name__ == "__main__":
    main()
