#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from ct_mcot.analysis.logs import summarize_train_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize CT-MCoT JSONL training logs.")
    parser.add_argument("--log", required=True)
    args = parser.parse_args()
    print(json.dumps(summarize_train_log(args.log), indent=2))


if __name__ == "__main__":
    main()
