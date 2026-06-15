from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PredictionRecord:
    id: str
    label: int | str
    prediction: int | str
    probabilities: list[float] = field(default_factory=list)
    answer_text: str | None = None
    rationale: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def write_predictions(records: list[PredictionRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def read_predictions(path: str | Path) -> list[PredictionRecord]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(PredictionRecord(**json.loads(line)))
    return records


def align_predictions(*paths: str | Path) -> list[list[PredictionRecord]]:
    groups = [read_predictions(path) for path in paths]
    ids = [[record.id for record in group] for group in groups]
    first = ids[0]
    for other in ids[1:]:
        if other != first:
            raise ValueError("Prediction files are not aligned by example id.")
    return groups
