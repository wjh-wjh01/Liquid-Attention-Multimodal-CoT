#!/usr/bin/env python3
from __future__ import annotations

import argparse

from ct_mcot.pipelines.matrix import generate_jobs, write_jobs_jsonl, write_jobs_shell


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate experiment matrix jobs.")
    parser.add_argument("--matrix", default="configs/experiment_matrix.yaml")
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--jsonl", default="outputs/matrix/jobs.jsonl")
    parser.add_argument("--shell", default="outputs/matrix/run_jobs.sh")
    args = parser.parse_args()
    jobs = generate_jobs(args.matrix, args.base_config)
    write_jobs_jsonl(jobs, args.jsonl)
    write_jobs_shell(jobs, args.shell)
    print(f"generated {len(jobs)} jobs")


if __name__ == "__main__":
    main()
