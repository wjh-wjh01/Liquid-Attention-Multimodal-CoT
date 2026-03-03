from __future__ import annotations  # 支持前向类型标注

from typing import Any, Dict, Optional

import torch
from torch import nn

from src.models.liquid_attention import LiquidAttention


class MultimodalCoTModel(nn.Module):
    def __init__(self, model_cfg: Dict[str, Any], default_flags: Dict[str, bool]) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.default_flags = dict(default_flags)

        hidden_dim = int(model_cfg["hidden_dim"])
        vocab_size = int(model_cfg["vocab_size"])
        image_dim = int(model_cfg["image_dim"])

        self.hidden_dim = hidden_dim

        # CoT 推理控制相关参数
        self.max_reasoning_steps = int(model_cfg.get("max_reasoning_steps", 5))
        self.min_reasoning_steps = int(model_cfg.get("min_reasoning_steps", 1))
        self.stop_threshold = float(model_cfg.get("stop_threshold", 0.75))
        self.attention_mode = str(model_cfg.get("attention_mode", "liquid"))

        # 文本 embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.dropout = nn.Dropout(float(model_cfg.get("dropout", 0.1)))

        # 图像特征映射到 hidden_dim
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # 文本上下文 + 图像表示 融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # 状态递推单元（逐步更新 reasoning state）
        self.state_update = nn.GRUCell(hidden_dim, hidden_dim)

        # 液态注意力模块
        self.attention = LiquidAttention(
            hidden_dim=hidden_dim,
            tau=float(model_cfg.get("tau", 0.5)),
            dt=float(model_cfg.get("dt", 0.2)),
            micro_steps=int(model_cfg.get("micro_steps", 4)),
        )

        # 停止预测头（控制是否继续推理）
        self.stop_head = nn.Linear(hidden_dim, 1)

        # 反思头（可选残差修正）
        self.reflection_head = nn.Linear(hidden_dim, hidden_dim)

        # 用于计算选项匹配分数
        self.choice_query = nn.Linear(hidden_dim, hidden_dim)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # 对 padding mask 做均值池化
        mask = mask.float().unsqueeze(-1)
        denom = torch.clamp(mask.sum(dim=1), min=1.0)
        return (x * mask).sum(dim=1) / denom

    def _merge_flags(self, flags: Optional[Dict[str, bool]]) -> Dict[str, bool]:
        # 运行时 flags 覆盖默认 flags
        merged = dict(self.default_flags)
        if flags:
            merged.update(flags)
        return merged

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        flags: Optional[Dict[str, bool]] = None
    ) -> Dict[str, torch.Tensor]:

        flags = self._merge_flags(flags)

        # ===== 取出 batch =====
        q_ids = batch["question_ids"]
        q_mask = batch["question_mask"]
        c_ids = batch["choice_ids"]
        c_mask = batch["choice_mask"]
        c_valid = batch["choice_valid"]
        image_feats = batch["image_feats"]

        # ===== 问题编码 =====
        q_emb = self.dropout(self.embedding(q_ids))
        q_repr = self._masked_mean(q_emb, q_mask)

        # ===== 图像编码 =====
        image_repr = self.image_encoder(image_feats)

        # 可关闭多模态
        if not flags.get("use_multimodal", True):
            image_repr = torch.zeros_like(image_repr)

        # 初始 reasoning state
        state = q_repr + image_repr

        # 推理步数控制
        steps = self.max_reasoning_steps if flags.get("use_cot_control", True) else 1

        prev_attn = None
        stop_probs = []
        entropies = []
        last_attn = None

        # ===== 多步推理循环 =====
        for step_idx in range(steps):

            attn, entropy = self.attention(
                token_states=q_emb,
                reasoning_state=state,
                prev_state=prev_attn,
                mode=self.attention_mode,
                use_ode=flags.get("use_ode", True),
                use_cross_step=flags.get("use_cross_step", True),
            )

            # attention 聚合 token
            context_tokens = torch.bmm(attn, q_emb)
            context = context_tokens.mean(dim=1)

            # 融合图像信息
            fused = self.fusion(torch.cat([context, image_repr], dim=-1))

            # 更新 state
            state = self.state_update(fused, state)

            # 可选 reflection 修正
            if flags.get("use_reflection", False):
                state = state + 0.1 * torch.tanh(self.reflection_head(state))

            # 预测是否停止
            stop_prob = torch.sigmoid(self.stop_head(state)).squeeze(-1)

            stop_probs.append(stop_prob)
            entropies.append(entropy)
            last_attn = attn

            # 是否跨步平滑
            if flags.get("use_cross_step", True):
                prev_attn = attn.detach()
            else:
                prev_attn = None

            # 推理早停（仅 eval 阶段）
            if (
                not self.training
                and flags.get("use_self_validation", True)
                and (step_idx + 1) >= self.min_reasoning_steps
                and float(stop_prob.mean().item()) >= self.stop_threshold
            ):
                break

        # ===== 选项编码 =====
        bsz, max_choices, max_len = c_ids.shape
        c_flat = c_ids.view(bsz * max_choices, max_len)
        c_mask_flat = c_mask.view(bsz * max_choices, max_len)

        c_emb = self.embedding(c_flat)
        c_repr = self._masked_mean(c_emb, c_mask_flat)
        c_repr = c_repr.view(bsz, max_choices, self.hidden_dim)

        # ===== 计算 logits =====
        query = self.choice_query(state).unsqueeze(1)
        logits = (c_repr * query).sum(dim=-1)

        # mask 无效选项
        logits = logits.masked_fill(~c_valid, -1e9)

        # ===== 整理输出 =====
        stop_probs_tensor = torch.stack(stop_probs, dim=1)
        entropy_tensor = torch.stack(entropies, dim=1)

        steps_used = torch.full(
            (bsz,),
            stop_probs_tensor.shape[1],
            device=logits.device,
            dtype=torch.long
        )

        return {
            "logits": logits,                          # 选项得分
            "stop_prob": stop_probs_tensor[:, -1],     # 最后一步停止概率
            "steps_used": steps_used,                  # 实际使用步数
            "attn_state": last_attn,                   # 最终注意力矩阵
            "attn_entropy": entropy_tensor.mean(dim=1),# 注意力熵（平均）
            "stop_probs": stop_probs_tensor,           # 每步停止概率
        }