#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import deep_update, load_yaml
from src.common.io_utils import write_json
from src.common.logging_utils import log_line
from src.common.experiments import parse_experiment_matrix
from src.training.pipeline import run_single_experiment


def _copy_alias_run(output_root: pathlib.Path, dataset: str, alias: str, target: str, seed: int) -> None:
    src = output_root / dataset / target / f"seed_{seed}"
    dst = output_root / dataset / alias / f"seed_{seed}"
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full experiment matrix")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--matrix", type=str, default="configs/experiments/experiment_matrix.yaml")
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)
    matrix_cfg = load_yaml(args.matrix)
    matrix = parse_experiment_matrix(matrix_cfg)
    output_root = pathlib.Path(base_cfg.get("output_dir", "outputs/runs"))

    all_results = []

    for dataset in matrix.datasets:
        log_line(f"=== dataset: {dataset} ===")
        for exp in matrix.experiments:
            cfg = deep_update(base_cfg, {"seed": matrix.seed})
            try:
                result = run_single_experiment(
                    run_cfg=cfg,
                    experiment_name=exp.name,
                    dataset_name=dataset,
                    flags_override=exp.flags_patch,
                    model_override=exp.model_patch,
                )
                all_results.append(result)
                log_line(
                    f"done {dataset}/{exp.name}: test_acc={result['test_accuracy']:.4f}, avg_steps={result['avg_steps']:.2f}"
                )
            except Exception as e:
                err = {
                    "dataset": dataset,
                    "experiment": exp.name,
                    "error": str(e),
                }
                all_results.append(err)
                log_line(f"failed {dataset}/{exp.name}: {e}")

        for alias, target in matrix.aliases.items():
            _copy_alias_run(output_root, dataset, alias, target, matrix.seed)
            log_line(f"alias created: {dataset}/{alias} -> {target}")

    write_json("outputs/reports/run_manifest.json", {"results": all_results})
    log_line("All experiments completed. Manifest: outputs/reports/run_manifest.json")


if __name__ == "__main__":
    main()
