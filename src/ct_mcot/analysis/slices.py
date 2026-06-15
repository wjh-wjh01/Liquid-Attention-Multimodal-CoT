from __future__ import annotations

from collections import defaultdict
from typing import Callable

from ct_mcot.evaluation.predictions import PredictionRecord, read_predictions


def slice_metrics(
    prediction_path: str,
    key_fn: Callable[[PredictionRecord], str] | None = None,
) -> list[dict]:
    records = read_predictions(prediction_path)
    if key_fn is None:
        key_fn = lambda record: str(record.metadata.get("category", record.metadata.get("benchmark", "all")))
    grouped: dict[str, list[PredictionRecord]] = defaultdict(list)
    for record in records:
        grouped[key_fn(record)].append(record)
    rows = []
    for key, items in grouped.items():
        correct = sum(int(str(item.prediction) == str(item.label)) for item in items)
        rows.append({"slice": key, "num_examples": len(items), "accuracy": correct / max(len(items), 1)})
    return sorted(rows, key=lambda row: row["slice"])
