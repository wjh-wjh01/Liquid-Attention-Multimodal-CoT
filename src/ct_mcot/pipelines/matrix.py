from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class ExperimentJob:
    name: str
    seed: int
    benchmark_config: str
    ablation_config: str | None
    output_dir: str
    command: list[str]


def generate_jobs(matrix_path: str | Path, base_config: str = "configs/base.yaml") -> list[ExperimentJob]:
    with Path(matrix_path).open("r", encoding="utf-8") as f:
        matrix = yaml.safe_load(f)["matrix"]
    jobs = []
    ablations = [None, *matrix.get("ablations", [])]
    for seed, benchmark, ablation in itertools.product(matrix["seeds"], matrix["benchmarks"], ablations):
        bench_name = Path(benchmark).stem
        ablation_name = Path(ablation).stem if ablation else "full"
        name = f"{bench_name}_{ablation_name}_seed{seed}"
        output_dir = f"outputs/matrix/{bench_name}/{ablation_name}/seed{seed}"
        command = [
            "python",
            "scripts/run_experiment.py",
            "--config",
            base_config,
            "--config",
            benchmark,
            "--set",
            f"train.seed={seed}",
            "--set",
            f"train.output_dir={output_dir}",
        ]
        if ablation:
            command.extend(["--config", ablation])
        jobs.append(ExperimentJob(name, seed, benchmark, ablation, output_dir, command))
    return jobs


def write_jobs_jsonl(jobs: list[ExperimentJob], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for job in jobs:
            f.write(json.dumps(asdict(job), ensure_ascii=False) + "\n")


def write_jobs_shell(jobs: list[ExperimentJob], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for job in jobs:
            f.write(f"echo '[job] {job.name}'\n")
            f.write(" ".join(_quote(part) for part in job.command) + "\n\n")


def _quote(value: str) -> str:
    if any(ch.isspace() for ch in value):
        return "'" + value.replace("'", "'\\''") + "'"
    return value
