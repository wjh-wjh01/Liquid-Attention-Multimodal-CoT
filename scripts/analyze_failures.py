#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from ct_mcot.analysis.failure import categorize_failures, failure_summary, write_failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Categorize incorrect predictions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    failures = categorize_failures(args.predictions)
    write_failures(failures, args.output)
    print(json.dumps(failure_summary(failures), indent=2))


if __name__ == "__main__":
    main()
