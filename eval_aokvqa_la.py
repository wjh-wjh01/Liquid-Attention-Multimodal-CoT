import os
import torch
from tqdm import tqdm
from datasets import load_dataset
from src.models.multimodal_cot import MultimodalCoTModel
from src.data.tokenizer import SimpleTokenizer

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def load_la_ode(checkpoint_dir, device):
    tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.json")
    tokenizer = SimpleTokenizer()
    tokenizer.load(tokenizer_path)
    
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
    model = MultimodalCoTModel(model_cfg=model_cfg, default_flags={"use_ode": True}).to(device)
    
    ckpt_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=device)["model_state"]
        model.load_state_dict(state_dict)
    model.eval()
    return model, tokenizer

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("加载 A-OKVQA 验证集（前 500 条）...")
    dataset = list(load_dataset("HuggingFaceM4/A-OKVQA", split="validation", streaming=True).take(500))
    
    checkpoint_dir = "/root/autodl-tmp/project/code/outputs/runs/scienceqa/liquid_full/seed_3407"
    model, tokenizer = load_la_ode(checkpoint_dir, device)
    
    print("\n开始评测 LA-ODE 在 A-OKVQA 上的表现...")
    correct = total = 0
    max_q_len = 256
    max_c_len = 64
    max_choices = 4
    
    for item in tqdm(dataset):
        try:
            question = item['question']
            choices = item['choices']
            answer_idx = item['correct_choice_idx']
            
            q_tokens = tokenizer.encode(question, max_q_len)
            q_mask = [1 if t != 0 else 0 for t in q_tokens]
            if len(q_tokens) < max_q_len:
                q_mask += [0] * (max_q_len - len(q_tokens))
                q_tokens += [0] * (max_q_len - len(q_tokens))
            
            c_ids, c_mask, c_valid = [], [], []
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
            
            batch = {
                "question_ids": torch.tensor([q_tokens[:max_q_len]], dtype=torch.long).to(device),
                "question_mask": torch.tensor([q_mask[:max_q_len]], dtype=torch.bool).to(device),
                "choice_ids": torch.tensor([c_ids], dtype=torch.long).to(device),
                "choice_mask": torch.tensor([c_mask], dtype=torch.bool).to(device),
                "choice_valid": torch.tensor([c_valid], dtype=torch.bool).to(device),
                "image_feats": torch.zeros((1, 64), dtype=torch.float32).to(device),
                "answer_idx": torch.tensor([answer_idx], dtype=torch.long).to(device)
            }
            
            with torch.no_grad():
                out = model(batch, flags={"use_ode": True, "use_cross_step": True})
                if torch.argmax(out["logits"], dim=-1).item() == answer_idx:
                    correct += 1
            total += 1
        except Exception:
            continue
            
    print(f"LA-ODE 准确率: {(correct/total)*100:.2f}%" if total > 0 else "评测失败")

if __name__ == "__main__":
    main()