#!/usr/bin/env python3
from __future__ import annotations

import argparse

from ct_mcot.training.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a CT-MCoT experiment with layered YAML configs.")
    parser.add_argument("--config", action="append", required=True, help="YAML config. Can be repeated.")
    parser.add_argument("--set", action="append", default=[], help="Override as dotted.key=value")
    args = parser.parse_args()
    run_experiment(args.config, args.set)


if __name__ == "__main__":
    main()
