from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CurriculumStage:
    index: int
    latent_segments: int
    rationale_weight: float
    grounding_weight: float
    superposition_weight: float


def build_curriculum(max_stage: int, rationale_available: bool = True) -> list[CurriculumStage]:
    stages = []
    for idx in range(max_stage + 1):
        frac = idx / max(max_stage, 1)
        stages.append(
            CurriculumStage(
                index=idx,
                latent_segments=idx,
                rationale_weight=(1.0 - frac) * 0.3 if rationale_available else 0.0,
                grounding_weight=0.1 + 0.2 * frac,
                superposition_weight=0.1 * frac,
            )
        )
    return stages
