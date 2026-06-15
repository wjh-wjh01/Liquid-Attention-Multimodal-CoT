#!/usr/bin/env python3
from __future__ import annotations

import argparse

from ct_mcot.features.export import export_manifest_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Export deterministic dry-run features from a benchmark manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    path = export_manifest_features(args.manifest, args.cache_root, seed=args.seed)
    print(path)


if __name__ == "__main__":
    main()
