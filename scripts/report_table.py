#!/usr/bin/env python3
from __future__ import annotations

import argparse

from ct_mcot.evaluation.reporting import aggregate_by_method, collect_metrics, write_markdown_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate metrics JSON files into a markdown table.")
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = aggregate_by_method(collect_metrics(args.metrics))
    write_markdown_table(rows, args.output)


if __name__ == "__main__":
    main()
