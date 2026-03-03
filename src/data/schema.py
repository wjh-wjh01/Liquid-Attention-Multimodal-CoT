from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class UnifiedSample:
    id: str
    dataset: str
    split: str
    question: str
    choices: List[str]
    answer_idx: int
    image_path: Optional[str]
    difficulty: Optional[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "dataset": self.dataset,
            "split": self.split,
            "question": self.question,
            "choices": self.choices,
            "answer_idx": int(self.answer_idx),
            "image_path": self.image_path,
            "difficulty": self.difficulty,
            "metadata": self.metadata,
        }
