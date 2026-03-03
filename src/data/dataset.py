from __future__ import annotations  # 允许在类型注解中使用尚未定义的类型（前向引用）

import io
import random
import glob
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel

from src.common.io_utils import read_jsonl
from src.data.image_features import image_to_feature
from src.data.tokenizer import SimpleTokenizer


@dataclass
class DatasetConfig:
    # 文本与图像相关的统一配置
    max_question_len: int      # 问题最大长度
    max_choice_len: int        # 选项最大长度
    image_dim: int             # 图像特征维度
    noise_prob: float = 0.0    # 文本噪声注入概率（用于数据增强）


def _inject_text_noise(text: str, noise_prob: float, rng: random.Random) -> str:
    # 简单字符级噪声注入：随机删除或替换为"#"
    if noise_prob <= 0.0 or not text:
        return text

    chars = list(text)
    out = []
    for ch in chars:
        if rng.random() < noise_prob:
            # 一半概率删除，一半概率替换
            if rng.random() < 0.5:
                continue
            out.append("#")
        else:
            out.append(ch)

    noisy = "".join(out).strip()
    # 避免全部被删导致空串
    return noisy if noisy else text


class JsonlReasoningDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        tokenizer: SimpleTokenizer,
        cfg: DatasetConfig,
        include_unlabeled: bool = False,
        seed: int = 3407,
    ) -> None:
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.image_cache = {}  # 用于缓存本地图片特征

        # 根据路径推断数据划分
        split_name = (
            "train" if "train" in jsonl_path
            else "val" if "val" in jsonl_path
            else "test"
        )

        # 若路径中未显式包含划分字段，默认按 train 处理
        if "train" not in jsonl_path and "val" not in jsonl_path and "test" not in jsonl_path:
            split_name = "train"

        search_name = "validation" if split_name == "val" else split_name

        # 优先尝试加载 parquet 格式（HF datasets）
        files = glob.glob(
            f"/root/autodl-tmp/project/data/raw/scienceqa_full/data/*{search_name}*.parquet"
        )

        self.use_hf = False
        if files:
            print(f"Loading data from parquet files: {files}")
            ds = load_dataset("parquet", data_files={search_name: files})[search_name]
            self.rows = ds
            self.use_hf = True
        else:
            # 若 parquet 不存在，回退到 jsonl
            print(f"Parquet files not found, falling back to jsonl: {jsonl_path}")
            self.rows = read_jsonl(jsonl_path)

            # 默认过滤掉未标注样本
            if not include_unlabeled:
                self.rows = [
                    r for r in self.rows
                    if int(r.get("answer_idx", -1)) >= 0
                ]

        # 初始化 CLIP，用于图像特征抽取
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(self.device)
        self.clip_model.eval()
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]

        # HF parquet 与 jsonl 字段略有不同
        if self.use_hf:
            question = str(r.get("question", ""))
            choices = [str(c) for c in r.get("choices", [])]
            ans_idx = int(r.get("answer", -1))
            img_obj = r.get("image", None)
        else:
            question = str(r.get("question", ""))
            choices = [str(c) for c in r.get("choices", [])]
            ans_idx = int(r.get("answer_idx", -1))
            img_obj = None

        # 可选文本扰动（仅训练时有意义）
        if self.cfg.noise_prob > 0.0:
            question = _inject_text_noise(question, self.cfg.noise_prob, self.rng)
            choices = [
                _inject_text_noise(c, self.cfg.noise_prob, self.rng)
                for c in choices
            ]

        return {
            "id": str(r.get("id", idx)),
            "question": question,
            "choices": choices,
            "answer_idx": ans_idx,
            "image_obj": img_obj,
            # parquet 中 image 已直接加载，因此不再使用 image_path
            "image_path": r.get("image_path") if not self.use_hf else None,
            "difficulty": r.get("difficulty", "normal"),
        }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 自定义 batch 拼接逻辑
        bsz = len(batch)
        max_choices = max(max(len(x["choices"]), 2) for x in batch)

        # 预分配张量
        q_ids = torch.zeros((bsz, self.cfg.max_question_len), dtype=torch.long)
        q_mask = torch.zeros((bsz, self.cfg.max_question_len), dtype=torch.bool)

        c_ids = torch.zeros((bsz, max_choices, self.cfg.max_choice_len), dtype=torch.long)
        c_mask = torch.zeros((bsz, max_choices, self.cfg.max_choice_len), dtype=torch.bool)
        choice_valid = torch.zeros((bsz, max_choices), dtype=torch.bool)

        labels = torch.full((bsz,), -1, dtype=torch.long)
        image_feats = torch.zeros((bsz, self.cfg.image_dim), dtype=torch.float32)

        ids, difficulties = [], []

        for i, row in enumerate(batch):
            ids.append(row["id"])
            difficulties.append(row["difficulty"])

            # 编码问题
            enc_q = self.tokenizer.encode(
                row["question"],
                self.cfg.max_question_len
            )
            q_ids[i] = torch.tensor(enc_q, dtype=torch.long)
            q_mask[i] = q_ids[i] != 0  # 0 视为 padding

            # 至少保证两个选项
            choices = row["choices"] if row["choices"] else ["Unknown", "Unknown"]

            for j, c in enumerate(choices[:max_choices]):
                enc_c = self.tokenizer.encode(
                    c,
                    self.cfg.max_choice_len
                )
                c_ids[i, j] = torch.tensor(enc_c, dtype=torch.long)
                c_mask[i, j] = c_ids[i, j] != 0
                choice_valid[i, j] = True

            labels[i] = int(row.get("answer_idx", -1))

            img_obj = row.get("image_obj")
            feat_dim = self.cfg.image_dim

            if img_obj is not None:
                try:
                    # HF parquet 中 image 可能以 dict(bytes=...) 形式存在
                    if isinstance(img_obj, dict) and "bytes" in img_obj:
                        img_obj = Image.open(io.BytesIO(img_obj["bytes"]))

                    # 使用 CLIP 提取图像特征
                    inputs = self.clip_processor(
                        images=img_obj.convert("RGB"),
                        return_tensors="pt"
                    ).to(self.device)

                    with torch.no_grad():
                        clip_features = self.clip_model.get_image_features(**inputs)

                    raw_feat = clip_features.cpu().numpy().flatten()

                    # 对齐到固定维度
                    if raw_feat.size >= feat_dim:
                        final_feat = raw_feat[:feat_dim]
                    else:
                        final_feat = np.zeros(feat_dim, dtype=np.float32)
                        final_feat[:raw_feat.size] = raw_feat

                    image_feats[i] = torch.tensor(final_feat, dtype=torch.float32)

                except Exception as e:
                    # 单样本失败不影响整体训练
                    print(f"Warning: Failed to extract image features for ID {row['id']}: {e}")
                    pass
            else:
                # 无直接图像对象时，走本地路径特征提取逻辑
                feat = image_to_feature(
                    row.get("image_path"),
                    feat_dim,
                    self.image_cache
                )
                image_feats[i] = torch.tensor(
                    feat[:feat_dim],
                    dtype=torch.float32
                )

        return {
            "id": ids,
            "question_ids": q_ids,
            "question_mask": q_mask,
            "choice_ids": c_ids,
            "choice_mask": c_mask,
            "choice_valid": choice_valid,
            "answer_idx": labels,
            "image_feats": image_feats,
            "difficulty": difficulties,
        }