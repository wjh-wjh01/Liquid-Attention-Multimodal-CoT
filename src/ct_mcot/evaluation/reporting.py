from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def collect_metrics(paths: list[str | Path]) -> list[dict]:
    rows = []
    for path in paths:
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data["metrics_path"] = str(path)
        rows.append(data)
    return rows


def aggregate_by_method(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("method", "ct_mcot"), row.get("benchmark", "unknown"))].append(row)
    output = []
    for (method, benchmark), items in grouped.items():
        acc = [float(x["accuracy"]) for x in items if "accuracy" in x]
        f1 = [float(x["macro_f1"]) for x in items if "macro_f1" in x]
        output.append(
            {
                "method": method,
                "benchmark": benchmark,
                "seeds": len(items),
                "accuracy_mean": sum(acc) / len(acc) if acc else None,
                "macro_f1_mean": sum(f1) / len(f1) if f1 else None,
            }
        )
    return output


def write_markdown_table(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["method", "benchmark", "seeds", "accuracy_mean", "macro_f1_mean"]
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(_fmt(row.get(h)) for h in headers) + " |\n")


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return "" if value is None else str(value)
