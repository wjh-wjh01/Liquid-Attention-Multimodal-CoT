#!/usr/bin/env python3
from __future__ import annotations

import argparse

from ct_mcot.datasets.preprocess import write_split_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert official benchmark files to CT-MCoT JSONL manifest.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-root", default=None)
    args = parser.parse_args()
    write_split_manifest(args.benchmark, args.source, args.output, args.image_root)


if __name__ == "__main__":
    main()
