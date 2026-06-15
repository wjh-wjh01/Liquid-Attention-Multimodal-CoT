from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass
class SyntheticConfig:
    num_examples: int = 1000
    num_tokens: int = 24
    input_dim: int = 128
    branching_factor: int = 3
    path_length: int = 3
    seed: int = 13


def make_synthetic_reachability(cfg: SyntheticConfig, output_path: str | Path) -> None:
    """Create a toy multimodal reachability split.

    Tokens are divided into text, visual, and knowledge groups. Positive examples
    contain a source-target path crossing at least two modality groups.
    """
    rng = random.Random(cfg.seed)
    torch_gen = torch.Generator().manual_seed(cfg.seed)
    rows = []
    for idx in range(cfg.num_examples):
        label = int(idx % 2 == 0)
        features = torch.randn(cfg.num_tokens, cfg.input_dim, generator=torch_gen) * 0.4
        source = rng.randrange(0, cfg.num_tokens // 3)
        target = rng.randrange(2 * cfg.num_tokens // 3, cfg.num_tokens)
        path = [source]
        cursor = source
        for hop in range(cfg.path_length - 1):
            low = min(cfg.num_tokens - 1, (hop + 1) * cfg.num_tokens // cfg.path_length)
            high = min(cfg.num_tokens - 1, low + cfg.branching_factor + 3)
            cursor = rng.randrange(low, high + 1)
            path.append(cursor)
        path.append(target)
        if label:
            signal = torch.randn(cfg.input_dim, generator=torch_gen)
            signal = signal / signal.norm().clamp_min(1e-6)
            for pos, node in enumerate(path):
                features[node] += signal * (2.0 - 0.15 * pos)
        else:
            distractors = rng.sample(range(cfg.num_tokens), k=min(len(path), cfg.num_tokens))
            for node in distractors:
                features[node] += torch.randn(cfg.input_dim, generator=torch_gen) * 0.3
        rows.append(
            {
                "id": f"syn-{cfg.seed}-{idx}",
                "features": features.tolist(),
                "mask": [1] * cfg.num_tokens,
                "label": label,
                "support_nodes": path if label else [],
            }
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class JsonlMemoryDataset(Dataset):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        with self.path.open("r", encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[idx]
        return {
            "id": row["id"],
            "memory": torch.tensor(row["features"], dtype=torch.float32),
            "mask": torch.tensor(row.get("mask", [1] * len(row["features"])), dtype=torch.bool),
            "label": torch.tensor(row["label"], dtype=torch.long),
        }


def collate_memory(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "id": [item["id"] for item in batch],
        "memory": torch.stack([item["memory"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
    }
