from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ct_mcot.evaluation.predictions import read_predictions


@dataclass
class FailureCase:
    id: str
    label: int | str
    prediction: int | str
    category: str
    confidence: float | None
    metadata: dict[str, Any]


def categorize_failures(prediction_path: str | Path) -> list[FailureCase]:
    failures = []
    for record in read_predictions(prediction_path):
        if str(record.label) == str(record.prediction):
            continue
        confidence = max(record.probabilities) if record.probabilities else None
        category = infer_failure_category(record.metadata, record.diagnostics)
        failures.append(
            FailureCase(
                id=record.id,
                label=record.label,
                prediction=record.prediction,
                category=category,
                confidence=confidence,
                metadata=record.metadata,
            )
        )
    return failures


def infer_failure_category(metadata: dict[str, Any], diagnostics: dict[str, Any]) -> str:
    if diagnostics.get("retrieval_empty") or diagnostics.get("retrieval_score", 1.0) < 0.1:
        return "retrieval_failure"
    if diagnostics.get("solver_hit_cap"):
        return "solver_cap"
    if diagnostics.get("visual_attention_mass", 1.0) < 0.05 and metadata.get("image_path"):
        return "weak_visual_grounding"
    if metadata.get("task") in {"geometry", "algebra", "chart"}:
        return "symbolic_or_arithmetic"
    if metadata.get("category") in {"physics", "chemistry", "biology"}:
        return "domain_knowledge"
    return "unclassified"


def write_failures(failures: list[FailureCase], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for failure in failures:
            f.write(json.dumps(asdict(failure), ensure_ascii=False) + "\n")


def failure_summary(failures: list[FailureCase]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for failure in failures:
        counts[failure.category] = counts.get(failure.category, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
