from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class FeatureRecord:
    id: str
    feature_path: str
    mask_path: str
    label: int | str
    shape: list[int]
    source_hash: str
    metadata: dict[str, Any]


class FeatureCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.feature_dir = self.root / "features"
        self.mask_dir = self.root / "masks"
        self.manifest_path = self.root / "manifest.jsonl"
        self.feature_dir.mkdir(parents=True, exist_ok=True)
        self.mask_dir.mkdir(parents=True, exist_ok=True)

    def put(
        self,
        example_id: str,
        features: torch.Tensor,
        mask: torch.Tensor,
        label: int | str,
        source: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> FeatureRecord:
        key = stable_hash({"id": example_id, **source})
        feature_path = self.feature_dir / f"{key}.pt"
        mask_path = self.mask_dir / f"{key}.pt"
        torch.save(features.cpu(), feature_path)
        torch.save(mask.cpu(), mask_path)
        record = FeatureRecord(
            id=example_id,
            feature_path=str(feature_path),
            mask_path=str(mask_path),
            label=label,
            shape=list(features.shape),
            source_hash=key,
            metadata=metadata or {},
        )
        with self.manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        return record

    def load_manifest(self) -> list[FeatureRecord]:
        if not self.manifest_path.exists():
            return []
        records = []
        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(FeatureRecord(**json.loads(line)))
        return records

    def get(self, record: FeatureRecord) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.load(record.feature_path, map_location="cpu"), torch.load(record.mask_path, map_location="cpu")


def stable_hash(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
