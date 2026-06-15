#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from ct_mcot.analysis.slices import slice_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute slice-level accuracy from prediction metadata.")
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()
    print(json.dumps(slice_metrics(args.predictions), indent=2))


if __name__ == "__main__":
    main()
