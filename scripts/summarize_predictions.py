#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from ct_mcot.metrics import summarize_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize prediction JSONL metrics.")
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()
    print(json.dumps(summarize_predictions(args.predictions), indent=2))


if __name__ == "__main__":
    main()
