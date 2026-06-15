#!/usr/bin/env python3
from __future__ import annotations

import argparse

from ct_mcot.pipelines.paper_pipeline import PaperPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run predefined CT-MCoT paper pipelines.")
    parser.add_argument("--name", choices=["synthetic"], default="synthetic")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.name == "synthetic":
        PaperPipeline.synthetic(args.repo_root, dry_run=args.dry_run).run()


if __name__ == "__main__":
    main()
