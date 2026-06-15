from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class EntityMention:
    text: str
    start: int
    end: int
    score: float


class EntityLinker:
    """Rule-based linker used as a deterministic fallback for artifact runs."""

    def __init__(self, min_len: int = 3):
        self.min_len = min_len
        self.pattern = re.compile(r"[A-Za-z][A-Za-z0-9_\- ]{2,}")

    def extract(self, text: str) -> list[EntityMention]:
        mentions = []
        for match in self.pattern.finditer(text):
            value = " ".join(match.group(0).split())
            if len(value) >= self.min_len:
                mentions.append(EntityMention(value, match.start(), match.end(), 1.0))
        return mentions
