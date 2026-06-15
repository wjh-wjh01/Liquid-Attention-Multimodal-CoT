from __future__ import annotations

import json
from pathlib import Path

import torch


def export_attention_heatmap(
    attention: torch.Tensor,
    tokens: list[str],
    path: str | Path,
    example_id: str,
) -> None:
    """Export attention data for external plotting libraries."""
    pooled = attention.mean(dim=0).detach().cpu()
    rows = []
    for slot in range(pooled.shape[0]):
        for token_idx in range(pooled.shape[1]):
            rows.append(
                {
                    "id": example_id,
                    "slot": slot,
                    "token_index": token_idx,
                    "token": tokens[token_idx] if token_idx < len(tokens) else f"tok_{token_idx}",
                    "attention": float(pooled[slot, token_idx]),
                }
            )
    _write_jsonl(rows, path)


def export_trajectory_summary(
    thought_states: list[torch.Tensor],
    attention_states: list[torch.Tensor],
    path: str | Path,
    example_id: str,
) -> None:
    rows = []
    for step, thought in enumerate(thought_states):
        rows.append(
            {
                "id": example_id,
                "step": step,
                "thought_norm": float(thought.norm(dim=-1).mean().detach().cpu()),
                "attention_norm": float(attention_states[step].norm(dim=-1).mean().detach().cpu())
                if step < len(attention_states)
                else None,
            }
        )
    _write_jsonl(rows, path)


def _write_jsonl(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
