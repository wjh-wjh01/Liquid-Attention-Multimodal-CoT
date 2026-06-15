from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .solvers import SolverConfig, solve


@dataclass
class CTMCoTConfig:
    input_dim: int = 128
    hidden_dim: int = 256
    thought_slots: int = 8
    num_classes: int = 2
    solver: str = "rk4"
    horizon: float = 1.0
    steps: int = 12
    tau_min: float = 0.05
    tau_max: float = 5.0
    diffusion_weight: float = 0.05


class CTMCoTModel(nn.Module):
    """Compact CT-MCoT block for reproducible experiments and synthetic diagnostics.

    The model expects precomputed multimodal memory tokens. For full VLM training,
    connect CLIP/LLM encoders upstream and feed their projected tokens here.
    """

    def __init__(self, cfg: CTMCoTConfig):
        super().__init__()
        self.cfg = cfg
        self.memory_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.bot = nn.Parameter(torch.randn(cfg.thought_slots, cfg.hidden_dim) * 0.02)
        self.init_mlp = nn.Sequential(
            nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.query = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.key = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.value = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.attn_residual = nn.Sequential(
            nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, 1),
        )
        self.thought_gate = nn.Sequential(
            nn.Linear(3 * cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.Sigmoid(),
        )
        self.thought_update = nn.Sequential(
            nn.Linear(3 * cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.tau_h = nn.Linear(3 * cfg.hidden_dim, cfg.hidden_dim)
        self.tau_s = nn.Linear(2 * cfg.hidden_dim, 1)
        self.answer_head = nn.Sequential(
            nn.LayerNorm(2 * cfg.hidden_dim),
            nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.num_classes),
        )

    def forward(
        self,
        memory: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        grid_laplacian: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        memory = self.memory_proj(memory)
        batch_size, num_tokens, _ = memory.shape
        pooled = self._masked_mean(memory, mask)
        h0 = self._init_thoughts(pooled)
        s0 = memory.new_zeros(batch_size, self.cfg.thought_slots, num_tokens)
        solver_cfg = SolverConfig(self.cfg.solver, self.cfg.horizon, self.cfg.steps)

        def field(_: float, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            return self._dynamics(state[0], state[1], memory, pooled, mask, grid_laplacian)

        ht, st = solve(field, (h0, s0), solver_cfg)
        thought_pool = ht.mean(dim=1)
        mem_pool = self._masked_mean(memory, mask)
        logits = self.answer_head(torch.cat([thought_pool, mem_pool], dim=-1))
        output = {"logits": logits, "thought": ht, "attention_logits": st}
        if return_diagnostics:
            output["attention"] = self._masked_softmax(st, mask)
            output["tau_h_mean"] = self._last_tau_h.mean()
            output["tau_s_mean"] = self._last_tau_s.mean()
        return output

    def _init_thoughts(self, pooled: torch.Tensor) -> torch.Tensor:
        bot = self.bot.unsqueeze(0).expand(pooled.shape[0], -1, -1)
        pooled_slots = pooled.unsqueeze(1).expand_as(bot)
        return self.init_mlp(torch.cat([bot, pooled_slots], dim=-1))

    def _dynamics(
        self,
        h: torch.Tensor,
        s: torch.Tensor,
        memory: torch.Tensor,
        pooled: torch.Tensor,
        mask: Optional[torch.Tensor],
        grid_laplacian: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.query(h)
        k = self.key(memory)
        drive = torch.matmul(q, k.transpose(1, 2)) / (self.cfg.hidden_dim**0.5)
        attn = self._masked_softmax(s, mask)
        evidence = torch.matmul(attn, self.value(memory))
        pooled_slots = pooled.unsqueeze(1).expand_as(h)

        pair = torch.cat(
            [
                h.unsqueeze(2).expand(-1, -1, memory.shape[1], -1),
                memory.unsqueeze(1).expand(-1, h.shape[1], -1, -1),
            ],
            dim=-1,
        )
        residual_s = self.attn_residual(pair).squeeze(-1)
        tau_s_input = torch.cat([evidence, h], dim=-1).unsqueeze(2).expand_as(pair)
        tau_s = self._bounded_tau(self.tau_s(tau_s_input).squeeze(-1))

        diffusion = 0.0
        if grid_laplacian is not None:
            diffusion = self.cfg.diffusion_weight * torch.matmul(s, grid_laplacian.transpose(0, 1))
        ds = (-s + drive + residual_s + diffusion) / tau_s

        thought_input = torch.cat([h, evidence, pooled_slots], dim=-1)
        gate = self.thought_gate(thought_input)
        proposal = self.thought_update(thought_input)
        target = (1.0 - gate) * h + gate * proposal
        tau_h = self._bounded_tau(self.tau_h(thought_input))
        dh = (-h + target) / tau_h
        self._last_tau_h = tau_h.detach()
        self._last_tau_s = tau_s.detach()
        return dh, ds

    def _bounded_tau(self, raw: torch.Tensor) -> torch.Tensor:
        tau = self.cfg.tau_min + F.softplus(raw)
        return torch.clamp(tau, max=self.cfg.tau_max)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=1)
        weights = mask.float().unsqueeze(-1)
        return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _masked_softmax(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(1), -1e9)
        return torch.softmax(logits, dim=-1)
