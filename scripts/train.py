#!/usr/bin/env python3
from __future__ import annotations

import argparse

from ct_mcot.train import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CT-MCoT from a YAML config.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train_from_config(args.config)


if __name__ == "__main__":
    main()
