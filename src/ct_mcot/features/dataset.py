from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .cache import FeatureCache, FeatureRecord


class CachedFeatureDataset(Dataset):
    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        self.cache = FeatureCache(self.manifest_path.parent)
        self.records = self.cache.load_manifest()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        features, mask = self.cache.get(record)
        return {
            "id": record.id,
            "memory": features.float(),
            "mask": mask.bool(),
            "label": torch.tensor(int(record.label), dtype=torch.long),
            "metadata": record.metadata,
        }


def collate_cached(batch: list[dict]) -> dict:
    max_len = max(item["memory"].shape[0] for item in batch)
    dim = batch[0]["memory"].shape[-1]
    memory = torch.zeros(len(batch), max_len, dim)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    labels = torch.stack([item["label"] for item in batch])
    ids = []
    metadata = []
    for i, item in enumerate(batch):
        n = item["memory"].shape[0]
        memory[i, :n] = item["memory"]
        mask[i, :n] = item["mask"]
        ids.append(item["id"])
        metadata.append(item["metadata"])
    return {"id": ids, "memory": memory, "mask": mask, "label": labels, "metadata": metadata}
