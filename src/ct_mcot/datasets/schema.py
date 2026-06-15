from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class BenchmarkExample:
    id: str
    question: str
    answer: str | int
    image_path: Optional[str] = None
    choices: list[str] = field(default_factory=list)
    rationale: Optional[str] = None
    ocr: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EncodedExample:
    id: str
    label: int
    text_tokens: list[int]
    image_path: Optional[str]
    choices: list[str]
    metadata: dict[str, Any]
