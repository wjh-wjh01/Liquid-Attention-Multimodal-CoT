from __future__ import annotations  # 允许在类型标注中使用前向引用

import math
from typing import Optional
from torchdiffeq import odeint
import torch
from torch import nn


class LiquidAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        tau: float = 0.5,
        dt: float = 0.2,
        micro_steps: int = 4
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tau = float(tau)            # 时间常数（控制收敛速度）
        self.dt = float(dt)              # 单步积分时长
        self.micro_steps = int(micro_steps)

        # 标准 Q/K 投影
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # reasoning state → context 映射
        self.ctx_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # 跨步平滑系数（EMA）
        self.mix_alpha = nn.Parameter(torch.tensor(0.5))

    class AttentionODEFunc(nn.Module):
        """
        注意力连续时间动力系统:
            dA/dt = (target - A) / tau
        """
        def __init__(self, tau, target):
            super().__init__()
            self.tau = max(tau, 1e-6)
            self.target = target

        def forward(self, t, attn_state):
            return (self.target - attn_state) / self.tau

    @staticmethod
    def _row_normalize(a: torch.Tensor) -> torch.Tensor:
        # 行归一化，避免数值不稳定
        a = torch.clamp(a, min=1e-7)
        return a / torch.clamp(a.sum(dim=-1, keepdim=True), min=1e-7)

    def _compute_base_scores(self, token_states: torch.Tensor) -> torch.Tensor:
        # 标准 scaled dot-product attention 分数
        q = self.q_proj(token_states)
        k = self.k_proj(token_states)
        return torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.hidden_dim)

    def _compute_target(
        self,
        token_states: torch.Tensor,
        reasoning_state: torch.Tensor
    ) -> torch.Tensor:
        # 基础 attention 分数
        base_scores = self._compute_base_scores(token_states)

        # reasoning state 融入 token 表示
        state_ctx = self.ctx_proj(reasoning_state).unsqueeze(1)
        token_ctx = torch.tanh(token_states + state_ctx)

        # context 相似度分数
        ctx_scores = torch.matmul(
            token_ctx,
            token_ctx.transpose(-1, -2)
        ) / math.sqrt(self.hidden_dim)

        # 目标 attention（0~1）
        return torch.sigmoid(base_scores + ctx_scores)

    def _dynamics(
        self,
        attn_state: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        # 离散形式下的一阶线性收敛动力学
        return (target - attn_state) / max(self.tau, 1e-6)

    def forward(
        self,
        token_states: torch.Tensor,
        reasoning_state: torch.Tensor,
        prev_state: Optional[torch.Tensor] = None,
        mode: str = "liquid",
        use_ode: bool = True,
        use_cross_step: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # 基础 attention
        base_scores = self._compute_base_scores(token_states)
        base_attn = torch.softmax(base_scores, dim=-1)

        # 跨 step 指数平滑
        if prev_state is not None and use_cross_step:
            alpha = torch.clamp(self.mix_alpha, 0.0, 1.0)
            attn = self._row_normalize(
                alpha * prev_state + (1.0 - alpha) * base_attn
            )
        else:
            attn = base_attn

        # 不启用连续动力学时直接返回
        if mode == "static" or not use_ode:
            final_attn = attn
        else:
            target = self._compute_target(token_states, reasoning_state)

            if mode == "discrete":
                # 单步欧拉更新
                final_attn = self._row_normalize(
                    attn + self.dt * self._dynamics(attn, target)
                )
            else:
                # 连续时间积分区间 [0, dt]
                integration_time = torch.tensor(
                    [0.0, self.dt],
                    device=attn.device
                )

                ode_func = self.AttentionODEFunc(self.tau, target)

                # 使用 rk4 数值积分
                trajectory = odeint(
                    ode_func,
                    attn,
                    integration_time,
                    method="rk4"
                )

                # 取最终时刻的状态
                final_attn = self._row_normalize(trajectory[-1])

        # 计算 attention 熵（可作为分析指标）
        entropy = -(
            final_attn *
            torch.log(torch.clamp(final_attn, min=1e-7))
        ).sum(dim=-1).mean(dim=-1)

        return final_attn, entropy