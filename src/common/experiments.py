from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ExperimentSpec:
    name: str
    group: str
    model_patch: Dict[str, Any]
    flags_patch: Dict[str, Any]


@dataclass
class ExperimentMatrix:
    seed: int
    datasets: List[str]
    experiments: List[ExperimentSpec]
    aliases: Dict[str, str]


def parse_experiment_matrix(cfg: Dict[str, Any]) -> ExperimentMatrix:
    experiments = []
    for item in cfg.get("experiments", []):
        experiments.append(
            ExperimentSpec(
                name=item["name"],
                group=item["group"],
                model_patch=item.get("model", {}),
                flags_patch=item.get("ablation_flags", {}),
            )
        )
    return ExperimentMatrix(
        seed=int(cfg.get("seed", 3407)),
        datasets=list(cfg.get("datasets", [])),
        experiments=experiments,
        aliases=dict(cfg.get("aliases", {})),
    )
