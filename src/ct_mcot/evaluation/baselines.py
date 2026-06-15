from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    reasoning_type: str
    uses_vision: bool
    uses_knowledge: bool
    continuous: bool
    notes: str


CONTROLLED_BASELINES = [
    BaselineSpec("no_cot", "direct", True, False, False, "Direct answer prediction."),
    BaselineSpec("text_cot", "discrete_text", True, False, False, "Text rationale before answer."),
    BaselineSpec("multimodal_cot", "discrete_multimodal", True, False, False, "Explicit multimodal rationale."),
    BaselineSpec("kam_cot", "knowledge_augmented_cot", True, True, False, "Knowledge-augmented CoT."),
    BaselineSpec("coconut_latent", "latent_recurrent", False, False, False, "Continuous thought for text-centric latent reasoning."),
    BaselineSpec("mcout_latent", "multimodal_latent", True, False, False, "Multimodal latent reasoning without CT attention logits."),
    BaselineSpec("neural_ode_transformer", "continuous_depth", True, False, True, "Generic ODE transformer baseline."),
]


def baseline_table() -> list[dict]:
    return [spec.__dict__ for spec in CONTROLLED_BASELINES]
