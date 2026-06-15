from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from ct_mcot.model import CTMCoTConfig, CTMCoTModel

from .heads import AnswerHead, RationaleDecoder
from .memory import MemoryBatch, MultimodalMemoryEncoder


@dataclass
class AdvancedCTMCoTConfig:
    text_dim: int = 1024
    vision_dim: int = 1024
    knowledge_dim: int = 1024
    hidden_dim: int = 1024
    thought_slots: int = 8
    num_classes: int = 4
    solver: str = "rk4"
    horizon: float = 1.0
    steps: int = 12
    tau_min: float = 0.05
    tau_max: float = 5.0
    rationale_vocab_size: int = 0


class AdvancedCTMCoT(nn.Module):
    """Full multimodal wrapper around the liquid CT-MCoT core."""

    def __init__(self, cfg: AdvancedCTMCoTConfig):
        super().__init__()
        self.cfg = cfg
        self.memory_encoder = MultimodalMemoryEncoder(
            text_dim=cfg.text_dim,
            vision_dim=cfg.vision_dim,
            knowledge_dim=cfg.knowledge_dim,
            hidden_dim=cfg.hidden_dim,
        )
        core_cfg = CTMCoTConfig(
            input_dim=cfg.hidden_dim,
            hidden_dim=cfg.hidden_dim,
            thought_slots=cfg.thought_slots,
            num_classes=cfg.num_classes,
            solver=cfg.solver,
            horizon=cfg.horizon,
            steps=cfg.steps,
            tau_min=cfg.tau_min,
            tau_max=cfg.tau_max,
        )
        self.core = CTMCoTModel(core_cfg)
        self.answer_head = AnswerHead(cfg.hidden_dim, cfg.num_classes)
        self.rationale = (
            RationaleDecoder(cfg.hidden_dim, cfg.rationale_vocab_size)
            if cfg.rationale_vocab_size > 0
            else None
        )

    def forward(
        self,
        batch: MemoryBatch,
        grid_laplacian: Optional[torch.Tensor] = None,
        rationale_targets: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        memory, mask, modality = self.memory_encoder(batch)
        output = self.core(memory, mask, grid_laplacian, return_diagnostics=return_diagnostics)
        output["logits"] = self.answer_head(output["thought"], memory, mask)
        output["memory"] = memory
        output["mask"] = mask
        output["modality"] = modality
        if self.rationale is not None:
            output["rationale_logits"] = self.rationale(output["thought"], rationale_targets)
        return output
