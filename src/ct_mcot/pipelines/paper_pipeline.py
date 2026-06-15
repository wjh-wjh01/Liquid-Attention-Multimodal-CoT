from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineStage:
    name: str
    command: list[str]
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    optional: bool = False


class PaperPipeline:
    def __init__(self, stages: list[PipelineStage], dry_run: bool = False):
        self.stages = stages
        self.dry_run = dry_run

    def run(self) -> None:
        for stage in self.stages:
            print(f"[pipeline] {stage.name}: {' '.join(stage.command)}")
            if self.dry_run:
                continue
            try:
                subprocess.run(stage.command, cwd=stage.cwd, env=stage.env or None, check=True)
            except subprocess.CalledProcessError:
                if stage.optional:
                    print(f"[pipeline] optional stage failed: {stage.name}")
                    continue
                raise

    @classmethod
    def synthetic(cls, repo_root: str | Path = ".", dry_run: bool = False) -> "PaperPipeline":
        root = str(repo_root)
        stages = [
            PipelineStage("generate_train", ["python", "scripts/make_synthetic.py", "--output", "data/synthetic/train.jsonl", "--num-examples", "1000", "--seed", "13"], cwd=root),
            PipelineStage("generate_test", ["python", "scripts/make_synthetic.py", "--output", "data/synthetic/test.jsonl", "--num-examples", "300", "--seed", "42"], cwd=root),
            PipelineStage("train", ["python", "scripts/train.py", "--config", "configs/synthetic.yaml"], cwd=root),
            PipelineStage("evaluate", ["python", "scripts/evaluate.py", "--config", "configs/synthetic.yaml"], cwd=root),
        ]
        return cls(stages, dry_run=dry_run)
