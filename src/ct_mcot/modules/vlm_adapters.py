from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class EncoderOutput:
    tokens: torch.Tensor
    mask: torch.Tensor
    metadata: dict[str, Any]


class LazyImportError(RuntimeError):
    pass


def _require_transformers():
    try:
        from transformers import AutoModel, AutoProcessor, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise LazyImportError(
            "transformers is required for VLM adapters. Install with: "
            "pip install -r requirements-vlm.txt"
        ) from exc
    return AutoModel, AutoProcessor, AutoTokenizer


class TextBackboneAdapter:
    def __init__(self, checkpoint: str, device: str = "cpu", trust_remote_code: bool = False):
        AutoModel, _, AutoTokenizer = _require_transformers()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=trust_remote_code)
        self.model = AutoModel.from_pretrained(checkpoint, trust_remote_code=trust_remote_code).to(device)
        self.device = torch.device(device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: list[str], max_length: int = 256) -> EncoderOutput:
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        batch = {k: v.to(self.device) for k, v in batch.items()}
        output = self.model(**batch)
        hidden = getattr(output, "last_hidden_state", output[0])
        return EncoderOutput(hidden.cpu(), batch["attention_mask"].bool().cpu(), {"checkpoint_device": str(self.device)})


class VisionBackboneAdapter:
    def __init__(self, checkpoint: str, device: str = "cpu"):
        AutoModel, AutoProcessor, _ = _require_transformers()
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint).to(device)
        self.device = torch.device(device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, image_paths: list[str | Path]) -> EncoderOutput:
        try:
            from PIL import Image
        except ImportError as exc:
            raise LazyImportError("Pillow is required for image encoding.") from exc
        images = [Image.open(path).convert("RGB") for path in image_paths]
        batch = self.processor(images=images, return_tensors="pt")
        batch = {k: v.to(self.device) for k, v in batch.items()}
        if hasattr(self.model, "vision_model"):
            output = self.model.vision_model(**batch)
        else:
            output = self.model(**batch)
        hidden = getattr(output, "last_hidden_state", output[0])
        mask = torch.ones(hidden.shape[:2], dtype=torch.bool)
        return EncoderOutput(hidden.cpu(), mask, {"num_images": len(image_paths)})


class KnowledgeTextEncoder:
    def __init__(self, text_adapter: TextBackboneAdapter):
        self.text_adapter = text_adapter

    def encode_nodes(self, node_texts: list[str], max_length: int = 64) -> EncoderOutput:
        encoded = self.text_adapter.encode(node_texts, max_length=max_length)
        pooled = encoded.tokens.mean(dim=1, keepdim=True)
        mask = torch.ones(pooled.shape[:2], dtype=torch.bool)
        return EncoderOutput(pooled, mask, encoded.metadata)
