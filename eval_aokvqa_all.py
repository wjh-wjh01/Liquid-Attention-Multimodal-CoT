import os
import torch
import gc
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from transformers import LlavaProcessor, LlavaForConditionalGeneration

# 导入自定义 LA-ODE 模型和分词器
from src.models.multimodal_cot import MultimodalCoTModel
from src.data.tokenizer import SimpleTokenizer

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def clear_vram():
    """清空显存，避免连续加载多个模型导致显存不足"""
    torch.cuda.empty_cache()
    gc.collect()

def load_la_ode(checkpoint_dir, device):
    """加载 LA-ODE 权重"""
    tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.json")
    tokenizer = SimpleTokenizer()
    tokenizer.load(tokenizer_path)
    
    # 模型尺寸参数（与 ScienceQA 训练配置保持一致）
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

def eval_la_ode(model, tokenizer, dataset, device):
    print("\n开始评测 LA-ODE 在 A-OKVQA 上的表现...")
    correct = total = 0
    max_q_len = 256
    max_c_len = 64
    max_choices = 4  # A-OKVQA 为 4 个选项
    
    for item in tqdm(dataset):
        try:
            question = item['question']
            choices = item['choices']
            answer_idx = item['correct_choice_idx']
            
            # 1. Question 处理
            q_tokens = tokenizer.encode(question, max_q_len)
            q_mask = [1 if t != 0 else 0 for t in q_tokens]
            if len(q_tokens) < max_q_len:
                q_mask += [0] * (max_q_len - len(q_tokens))
                q_tokens += [0] * (max_q_len - len(q_tokens))
            
            # 2. Choices 处理
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
                "image_feats": torch.zeros((1, 64), dtype=torch.float32).to(device),  # 不使用视觉特征，仅测试推理逻辑
                "answer_idx": torch.tensor([answer_idx], dtype=torch.long).to(device)
            }
            
            with torch.no_grad():
                out = model(batch, flags={"use_ode": True, "use_cross_step": True})
                pred_idx = torch.argmax(out["logits"], dim=-1).item()
                if pred_idx == answer_idx:
                    correct += 1
            total += 1
        except Exception:
            continue
            
    print(f"LA-ODE 准确率: {(correct/total)*100:.2f}%" if total > 0 else "评测失败")

def eval_baseline(model_name, model, processor, dataset, device):
    print(f"\n开始评测 {model_name} 在 A-OKVQA 上的表现...")
    correct = total = 0
    for item in tqdm(dataset):
        try:
            question = item['question']
            choices = item['choices']
            correct_idx = item['correct_choice_idx']
            image = item['image'].convert('RGB')
            
            choices_str = " ".join([f"({chr(65+j)}) {c}" for j, c in enumerate(choices)])
            prompt = f"Question: {question}\nChoices: {choices_str}\nAnswer with the correct choice directly."
            
            if "qwen" in model_name.lower():
                messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(device)
            else:
                prompt_text = f"USER: <image>\n{prompt}\nASSISTANT:"
                inputs = processor(text=prompt_text, images=image, return_tensors="pt").to(device, torch.float16)
                
            input_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=10, temperature=0.1, do_sample=False)
            
            pred_text = processor.decode(output_ids[0][input_len:], skip_special_tokens=True).lower().strip()
            correct_ans = str(choices[correct_idx]).lower()
            
            if correct_ans in pred_text or chr(65+correct_idx).lower() in pred_text:
                correct += 1
            total += 1
        except Exception:
            continue
            
    print(f"{model_name} 准确率: {(correct/total)*100:.2f}%" if total > 0 else "评测失败")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("正在加载 A-OKVQA 验证集（前 500 条样本）...")
    dataset_iter = load_dataset("HuggingFaceM4/A-OKVQA", split="validation", streaming=True).take(500)
    dataset = list(dataset_iter)
    
    # 1. 评测 LA-ODE
    checkpoint_dir = "/root/autodl-tmp/project/code/outputs/runs/scienceqa/liquid_full/seed_3407"
    la_model, tokenizer = load_la_ode(checkpoint_dir, device)
    eval_la_ode(la_model, tokenizer, dataset, device)
    del la_model, tokenizer
    clear_vram()
    
    # 2. 评测 Qwen2-VL
    qwen_id = "Qwen/Qwen2-VL-2B-Instruct"
    qwen_processor = AutoProcessor.from_pretrained(qwen_id)
    qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(qwen_id, torch_dtype=torch.float16, device_map=device)
    eval_baseline("Qwen2-VL-2B", qwen_model, qwen_processor, dataset, device)
    del qwen_model, qwen_processor
    clear_vram()
    
    # 3. 评测 LLaVA-1.5
    llava_id = "llava-hf/llava-1.5-7b-hf"
    llava_processor = LlavaProcessor.from_pretrained(llava_id)
    llava_model = LlavaForConditionalGeneration.from_pretrained(llava_id, torch_dtype=torch.float16, device_map=device)
    eval_baseline("LLaVA-1.5-7B", llava_model, llava_processor, dataset, device)
    del llava_model, llava_processor
    clear_vram()
    
    print("\nA-OKVQA 测试流程结束。")

if __name__ == "__main__":
    main()