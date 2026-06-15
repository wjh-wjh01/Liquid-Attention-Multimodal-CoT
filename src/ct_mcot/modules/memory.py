from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class MemoryBatch:
    text: Optional[torch.Tensor] = None
    vision: Optional[torch.Tensor] = None
    knowledge: Optional[torch.Tensor] = None
    text_mask: Optional[torch.Tensor] = None
    vision_mask: Optional[torch.Tensor] = None
    knowledge_mask: Optional[torch.Tensor] = None


class MultimodalMemoryEncoder(nn.Module):
    """Project text, visual, and knowledge tokens into one shared memory bank."""

    def __init__(
        self,
        text_dim: int,
        vision_dim: int,
        knowledge_dim: int,
        hidden_dim: int,
        max_tokens: int = 4096,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.knowledge_proj = nn.Linear(knowledge_dim, hidden_dim)
        self.modality_embed = nn.Embedding(3, hidden_dim)
        self.position_embed = nn.Embedding(max_tokens, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, batch: MemoryBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chunks: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        modality_ids: list[torch.Tensor] = []
        device = self._infer_device(batch)

        if batch.text is not None:
            chunks.append(self.text_proj(batch.text))
            masks.append(self._mask_or_ones(batch.text, batch.text_mask))
            modality_ids.append(torch.zeros(batch.text.shape[:2], dtype=torch.long, device=device))
        if batch.vision is not None:
            chunks.append(self.vision_proj(batch.vision))
            masks.append(self._mask_or_ones(batch.vision, batch.vision_mask))
            modality_ids.append(torch.ones(batch.vision.shape[:2], dtype=torch.long, device=device))
        if batch.knowledge is not None:
            chunks.append(self.knowledge_proj(batch.knowledge))
            masks.append(self._mask_or_ones(batch.knowledge, batch.knowledge_mask))
            modality_ids.append(torch.full(batch.knowledge.shape[:2], 2, dtype=torch.long, device=device))
        if not chunks:
            raise ValueError("At least one modality must be present.")

        memory = torch.cat(chunks, dim=1)
        mask = torch.cat(masks, dim=1)
        modality = torch.cat(modality_ids, dim=1)
        pos = torch.arange(memory.shape[1], device=memory.device).unsqueeze(0).expand(memory.shape[0], -1)
        memory = memory + self.modality_embed(modality) + self.position_embed(pos)
        return self.norm(memory), mask, modality

    @staticmethod
    def _infer_device(batch: MemoryBatch) -> torch.device:
        for value in (batch.text, batch.vision, batch.knowledge):
            if value is not None:
                return value.device
        return torch.device("cpu")

    @staticmethod
    def _mask_or_ones(tokens: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is not None:
            return mask.bool()
        return torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
