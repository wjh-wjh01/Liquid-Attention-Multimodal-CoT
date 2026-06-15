#!/usr/bin/env python3
from __future__ import annotations

import argparse

from ct_mcot.evaluate import evaluate_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CT-MCoT from a YAML config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    evaluate_from_config(args.config, args.checkpoint)


if __name__ == "__main__":
    main()
