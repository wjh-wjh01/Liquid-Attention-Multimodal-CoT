#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_yaml
from src.common.io_utils import write_json
from src.common.logging_utils import log_line
from src.data.adapters import prepare_all_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local datasets into unified JSONL format")
    parser.add_argument("--config", type=str, default="configs/datasets/local_v1.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    raw_root = pathlib.Path(cfg.get("raw_root", "data/raw"))
    processed_root = pathlib.Path(cfg.get("processed_root", "data/processed"))
    subsets = cfg.get("subsets", {})
    seed = int(cfg.get("seed", 3407))

    reports = prepare_all_datasets(
        raw_root=raw_root,
        processed_root=processed_root,
        subsets_cfg=subsets,
        seed=seed,
    )

    missing = [r for r in reports if r.get("status") != "ok"]
    summary = {
        "profile": cfg.get("profile", "local_v1"),
        "seed": seed,
        "reports": reports,
        "missing_count": len(missing),
    }

    write_json("outputs/reports/prepare_data_report.json", summary)
    write_json("outputs/reports/missing_data_report.json", {"missing": missing})

    for r in reports:
        log_line(f"{r['dataset']} status={r['status']} counts={r.get('counts', {})}")
        for w in r.get("warnings", []):
            log_line(f"  warning: {w}")

    if missing:
        log_line("Some datasets are missing local raw files. See outputs/reports/missing_data_report.json")
    else:
        log_line("All configured datasets prepared successfully.")


if __name__ == "__main__":
    main()
