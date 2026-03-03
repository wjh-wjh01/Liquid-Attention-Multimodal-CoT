import os
import torch
import json
from tqdm import tqdm
from datasets import load_dataset

from src.models.multimodal_cot import MultimodalCoTModel
from src.data.tokenizer import SimpleTokenizer

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def load_custom_model(checkpoint_dir, device):
    """加载训练好的 LA-ODE 权重和词表"""
    # 1. 加载词表
    tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.json")
    tokenizer = SimpleTokenizer()
    tokenizer.load(tokenizer_path)

    # 2. 模型配置需与训练 ScienceQA 时保持一致
    model_cfg = {
        "hidden_dim": 128,
        "vocab_size": 20000,
        "image_dim": 64,
        "max_reasoning_steps": 5,
        "min_reasoning_steps": 1,
        "stop_threshold": 0.75,
        "attention_mode": "liquid",
        "dropout": 0.1,
        "tau": 0.5,
        "dt": 0.2,
        "micro_steps": 4
    }

    model = MultimodalCoTModel(
        model_cfg=model_cfg,
        default_flags={"use_ode": True}
    ).to(device)

    ckpt_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=device)["model_state"]
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("checkpoint.pt not found. Using randomly initialized weights.")

    model.eval()
    return model, tokenizer


def evaluate_mmlu_pro(model, tokenizer, device):
    print("\nLoading MMLU-Pro (first 500 samples)...")
    dataset = load_dataset(
        "TIGER-Lab/MMLU-Pro",
        split="test",
        streaming=True
    ).take(500)

    correct = 0
    total = 0

    max_q_len = 256
    max_c_len = 64
    max_choices = 10

    for item in tqdm(dataset, total=500):
        question = item.get("question", "")
        choices = item.get("options", item.get("choices", []))

        if "answer_index" in item:
            answer_idx = item["answer_index"]
        else:
            ans = item.get("answer", 0)
            answer_idx = ord(ans.upper()) - 65 if isinstance(ans, str) else ans

        # 1. 处理 Question
        q_tokens = tokenizer.encode(question, max_q_len)
        q_mask = [1 if t != 0 else 0 for t in q_tokens]
        if len(q_tokens) < max_q_len:
            q_mask += [0] * (max_q_len - len(q_tokens))
            q_tokens += [0] * (max_q_len - len(q_tokens))
        q_tokens = q_tokens[:max_q_len]
        q_mask = q_mask[:max_q_len]

        # 2. 处理 Choices
        c_ids = []
        c_mask = []
        c_valid = []

        for i in range(max_choices):
            if i < len(choices):
                c_toks = tokenizer.encode(str(choices[i]), max_c_len)
                c_m = [1 if t != 0 else 0 for t in c_toks]
                if len(c_toks) < max_c_len:
                    c_m += [0] * (max_c_len - len(c_toks))
                    c_toks += [0] * (max_c_len - len(c_toks))
                c_ids.append(c_toks[:max_c_len])
                c_mask.append(c_m[:max_c_len])
                c_valid.append(True)
            else:
                c_ids.append([0] * max_c_len)
                c_mask.append([0] * max_c_len)
                c_valid.append(False)

        # 3. 构建 Batch
        batch = {
            "question_ids": torch.tensor([q_tokens], dtype=torch.long).to(device),
            "question_mask": torch.tensor([q_mask], dtype=torch.bool).to(device),
            "choice_ids": torch.tensor([c_ids], dtype=torch.long).to(device),
            "choice_mask": torch.tensor([c_mask], dtype=torch.bool).to(device),
            "choice_valid": torch.tensor([c_valid], dtype=torch.bool).to(device),
            "image_feats": torch.zeros((1, 64), dtype=torch.float32).to(device),
            "answer_idx": torch.tensor([answer_idx], dtype=torch.long).to(device)
        }

        # 4. 前向推理
        with torch.no_grad():
            out = model(
                batch,
                flags={
                    "use_ode": True,
                    "use_cross_step": True,
                    "use_self_validation": True
                }
            )
            logits = out["logits"]
            pred_idx = torch.argmax(logits, dim=-1).item()

            if pred_idx == answer_idx:
                correct += 1
            total += 1

    acc = correct / total if total > 0 else 0
    print(f"\nEvaluation finished. Zero-shot accuracy on MMLU-Pro: {acc*100:.2f}%")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_directory = "/root/autodl-tmp/project/code/outputs/runs/scienceqa/liquid_full/seed_3407"

    model, tokenizer = load_custom_model(checkpoint_directory, device)
    evaluate_mmlu_pro(model, tokenizer, device)