#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
from typing import Any, Dict, List
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_yaml
from src.common.io_utils import read_json, write_csv, write_json


def _safe_read(path: pathlib.Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return read_json(path)
    except Exception:
        return {}


def _to_markdown_table(rows: List[Dict[str, Any]], headers: List[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect experiment outputs into report tables")
    parser.add_argument("--runs-dir", type=str, default="outputs/runs")
    parser.add_argument("--output-dir", type=str, default="outputs/reports")
    parser.add_argument("--matrix", type=str, default="configs/experiments/experiment_matrix.yaml")
    args = parser.parse_args()

    runs_dir = pathlib.Path(args.runs_dir)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir.parent / "figures").mkdir(parents=True, exist_ok=True)

    matrix = load_yaml(args.matrix)
    datasets = matrix.get("datasets", ["scienceqa", "mmlu_pro", "cmmcot"])
    aliases = matrix.get("aliases", {})

    comparison_methods = [
        "static_cot",
        "discrete_evolving_attention",
        "reflective_cot",
        "static_self_validation",
        "liquid_full",
    ]
    ablation_methods = [
        "w_o_ode",
        "w_o_cross_step",
        "w_o_self_validation",
        "w_o_multimodal",
        "w_o_cot_control",
    ]

    eval_rows: Dict[str, Dict[str, Any]] = {}
    eff_rows: List[Dict[str, Any]] = []
    rob_rows: List[Dict[str, Any]] = []

    for ds_dir in sorted(runs_dir.glob("*")):
        if not ds_dir.is_dir():
            continue
        dataset = ds_dir.name
        for exp_dir in sorted(ds_dir.glob("*")):
            exp_name = exp_dir.name
            for seed_dir in sorted(exp_dir.glob("seed_*")):
                eval_json = _safe_read(seed_dir / "eval_metrics.json")
                eff_json = _safe_read(seed_dir / "efficiency.json")
                rob_json = _safe_read(seed_dir / "robustness_metrics.json")

                key = f"{dataset}::{exp_name}"
                eval_rows[key] = {
                    "dataset": dataset,
                    "experiment": exp_name,
                    "accuracy": float(eval_json.get("test", {}).get("accuracy", 0.0)),
                }

                if eff_json:
                    eff_rows.append(
                        {
                            **eff_json,
                            "dataset": dataset,
                            "experiment": exp_name,
                        }
                    )
                if rob_json:
                    rob_rows.append({"dataset": dataset, "experiment": exp_name, **rob_json})

    for alias, target in aliases.items():
        for ds in datasets:
            src_key = f"{ds}::{target}"
            if src_key not in eval_rows:
                continue
            eval_rows[f"{ds}::{alias}"] = {
                "dataset": ds,
                "experiment": alias,
                "accuracy": eval_rows[src_key]["accuracy"],
            }

    def build_table(methods: List[str]) -> List[Dict[str, Any]]:
        rows = []
        for method in methods:
            row = {"method": method}
            vals = []
            for ds in datasets:
                acc = eval_rows.get(f"{ds}::{method}", {}).get("accuracy", 0.0)
                pct = round(acc * 100.0, 2)
                row[ds] = pct
                vals.append(pct)
            row["avg"] = round(sum(vals) / max(len(vals), 1), 2)
            rows.append(row)
        return rows

    comparison = build_table(comparison_methods)
    ablation = build_table(ablation_methods)

    write_csv(out_dir / "table_comparison.csv", comparison, ["method", *datasets, "avg"])
    write_csv(out_dir / "table_ablation.csv", ablation, ["method", *datasets, "avg"])

    with (out_dir / "table_comparison.md").open("w", encoding="utf-8") as f:
        f.write(_to_markdown_table(comparison, ["method", *datasets, "avg"]))
    with (out_dir / "table_ablation.md").open("w", encoding="utf-8") as f:
        f.write(_to_markdown_table(ablation, ["method", *datasets, "avg"]))

    if eff_rows:
        keys = ["dataset", "experiment", "seed", "train_time_sec", "eval_time_sec", "peak_memory_mb", "avg_steps", "accuracy"]
        write_csv(out_dir / "efficiency.csv", eff_rows, keys)

    if rob_rows:
        rob_keys = ["dataset", "experiment", "simple_acc", "difficulty_acc", "noise_10_acc", "noise_20_acc"]
        for r in rob_rows:
            for k in rob_keys:
                r.setdefault(k, 0.0)
        write_csv(out_dir / "robustness.csv", rob_rows, rob_keys)

    # Figure 1: comparison bars
    x = range(len(comparison_methods))
    width = 0.25
    plt.figure(figsize=(10, 4))
    for i, ds in enumerate(datasets):
        vals = [r.get(ds, 0.0) for r in comparison]
        plt.bar([p + i * width for p in x], vals, width=width, label=ds)
    plt.xticks([p + width for p in x], comparison_methods, rotation=20)
    plt.ylabel("Accuracy (%)")
    plt.title("Comparison Experiments")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir.parent / "figures" / "comparison_bars.png", dpi=180)
    plt.close()

    # Figure 2: robustness for liquid_full
    liquid_rows = [r for r in rob_rows if r.get("experiment") == "liquid_full"]
    if liquid_rows:
        liquid_rows = sorted(liquid_rows, key=lambda r: datasets.index(r["dataset"]) if r["dataset"] in datasets else 999)
        dnames = [r["dataset"] for r in liquid_rows]
        clean = [eval_rows.get(f"{d}::liquid_full", {}).get("accuracy", 0.0) * 100.0 for d in dnames]
        n10 = [r.get("noise_10_acc", 0.0) * 100.0 for r in liquid_rows]
        n20 = [r.get("noise_20_acc", 0.0) * 100.0 for r in liquid_rows]
        plt.figure(figsize=(8, 4))
        plt.plot(dnames, clean, marker="o", label="clean")
        plt.plot(dnames, n10, marker="o", label="noise 10%")
        plt.plot(dnames, n20, marker="o", label="noise 20%")
        plt.ylabel("Accuracy (%)")
        plt.title("Robustness of liquid_full")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir.parent / "figures" / "robustness_liquid_full.png", dpi=180)
        plt.close()

    write_json(
        out_dir / "report_summary.json",
        {
            "comparison_rows": len(comparison),
            "ablation_rows": len(ablation),
            "efficiency_rows": len(eff_rows),
            "robustness_rows": len(rob_rows),
        },
    )

    print({"comparison": str(out_dir / "table_comparison.csv"), "ablation": str(out_dir / "table_ablation.csv")})


if __name__ == "__main__":
    main()
