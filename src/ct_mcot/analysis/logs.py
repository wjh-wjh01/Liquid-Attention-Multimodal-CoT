from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize_train_log(path: str | Path) -> dict:
    rows = read_jsonl(path)
    by_event: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_event[row.get("event", "unknown")].append(row)
    last_step = by_event.get("step", [{}])[-1]
    last_epoch = by_event.get("epoch", [{}])[-1]
    return {
        "num_step_events": len(by_event.get("step", [])),
        "num_epoch_events": len(by_event.get("epoch", [])),
        "last_step_metrics": last_step.get("metrics", {}),
        "last_epoch_metrics": last_epoch.get("metrics", {}),
        "elapsed_sec": max([float(row.get("elapsed_sec", 0.0)) for row in rows], default=0.0),
    }


def collect_failure_logs(paths: list[str | Path]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in paths:
        for row in read_jsonl(path):
            category = row.get("category", "unknown")
            counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
