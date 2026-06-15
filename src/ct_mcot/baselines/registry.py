from __future__ import annotations

from torch import nn

from .direct import DirectPredictionModel
from .latent_recurrent import LatentRecurrentModel
from .ode_transformer import ODETransformerBaseline


def build_baseline(name: str, cfg: dict) -> nn.Module:
    name = name.lower()
    if name in {"no_cot", "direct"}:
        return DirectPredictionModel(
            input_dim=cfg["input_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_classes=cfg["num_classes"],
            pooling=cfg.get("pooling", "mean"),
        )
    if name in {"latent_recurrent", "mcout", "coconut"}:
        return LatentRecurrentModel(
            input_dim=cfg["input_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_classes=cfg["num_classes"],
            thought_slots=cfg.get("thought_slots", 8),
            recurrent_steps=cfg.get("recurrent_steps", cfg.get("steps", 12)),
        )
    if name in {"ode_transformer", "neural_ode_transformer"}:
        return ODETransformerBaseline(
            input_dim=cfg["input_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_classes=cfg["num_classes"],
            num_heads=cfg.get("num_heads", 4),
            steps=cfg.get("steps", 12),
        )
    raise ValueError(f"Unsupported baseline: {name}")
