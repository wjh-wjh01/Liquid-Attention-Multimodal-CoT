from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schema import BenchmarkExample


def load_benchmark(name: str, path: str | Path) -> list[BenchmarkExample]:
    name = name.lower()
    if name == "scienceqa":
        return list(load_scienceqa(path))
    if name in {"a-okvqa", "aokvqa"}:
        return list(load_aokvqa(path))
    if name == "mmmu":
        return list(load_mmmu(path))
    if name == "mmstar":
        return list(load_mmstar(path))
    if name == "mathvista":
        return list(load_mathvista(path))
    raise ValueError(f"Unsupported benchmark adapter: {name}")


def load_json_or_jsonl(path: str | Path) -> Iterable[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            for line in f:
                if line.strip():
                    yield json.loads(line)
        else:
            data = json.load(f)
            rows = data.values() if isinstance(data, dict) else data
            yield from rows


def load_scienceqa(path: str | Path) -> Iterable[BenchmarkExample]:
    for row in load_json_or_jsonl(path):
        yield BenchmarkExample(
            id=str(row.get("id", row.get("qid"))),
            question=row.get("question", ""),
            choices=row.get("choices", row.get("options", [])),
            answer=row.get("answer", row.get("label")),
            image_path=row.get("image", row.get("image_path")),
            rationale=row.get("lecture") or row.get("solution") or row.get("rationale"),
            metadata={"subject": row.get("subject"), "topic": row.get("topic")},
        )


def load_aokvqa(path: str | Path) -> Iterable[BenchmarkExample]:
    for row in load_json_or_jsonl(path):
        yield BenchmarkExample(
            id=str(row.get("question_id", row.get("id"))),
            question=row.get("question", ""),
            choices=row.get("choices", []),
            answer=row.get("correct_choice_idx", row.get("answer")),
            image_path=row.get("image_path") or row.get("image"),
            rationale=row.get("rationales", [None])[0] if row.get("rationales") else None,
            metadata={"direct_answers": row.get("direct_answers", [])},
        )


def load_mmmu(path: str | Path) -> Iterable[BenchmarkExample]:
    for row in load_json_or_jsonl(path):
        yield BenchmarkExample(
            id=str(row.get("id")),
            question=row.get("question", row.get("problem", "")),
            choices=row.get("options", row.get("choices", [])),
            answer=row.get("answer"),
            image_path=row.get("image_path") or row.get("image"),
            metadata={"category": row.get("category"), "subfield": row.get("subfield")},
        )


def load_mmstar(path: str | Path) -> Iterable[BenchmarkExample]:
    for row in load_json_or_jsonl(path):
        yield BenchmarkExample(
            id=str(row.get("id", row.get("question_id"))),
            question=row.get("question", ""),
            choices=row.get("options", row.get("choices", [])),
            answer=row.get("answer"),
            image_path=row.get("image") or row.get("image_path"),
            metadata={"l2_category": row.get("l2_category"), "visual_dependency": row.get("visual_dependency")},
        )


def load_mathvista(path: str | Path) -> Iterable[BenchmarkExample]:
    for row in load_json_or_jsonl(path):
        yield BenchmarkExample(
            id=str(row.get("pid", row.get("id"))),
            question=row.get("query", row.get("question", "")),
            choices=row.get("choices", []),
            answer=row.get("answer"),
            image_path=row.get("image") or row.get("image_path"),
            metadata={"task": row.get("task"), "skills": row.get("skills", [])},
        )
