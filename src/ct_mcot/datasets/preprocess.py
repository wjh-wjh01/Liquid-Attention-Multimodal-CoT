from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .adapters import load_benchmark
from .normalization import choice_to_index


def write_split_manifest(
    benchmark: str,
    source_path: str | Path,
    output_path: str | Path,
    image_root: str | Path | None = None,
) -> None:
    examples = load_benchmark(benchmark, source_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            image_path = ex.image_path
            if image_path and image_root and not Path(image_path).is_absolute():
                image_path = str(Path(image_root) / image_path)
            label = choice_to_index(ex.answer, ex.choices) if ex.choices else ex.answer
            row = {
                "id": ex.id,
                "benchmark": benchmark,
                "question": ex.question,
                "choices": ex.choices,
                "label": label,
                "answer": ex.answer,
                "image_path": image_path,
                "rationale": ex.rationale,
                "ocr": ex.ocr,
                "metadata": ex.metadata,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def map_manifest(
    input_path: str | Path,
    output_path: str | Path,
    fn: Callable[[dict], dict],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(input_path).open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if line.strip():
                dst.write(json.dumps(fn(json.loads(line)), ensure_ascii=False) + "\n")
