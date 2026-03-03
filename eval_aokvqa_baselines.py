import os
import torch
import gc
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from transformers import LlavaProcessor, LlavaForConditionalGeneration

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def clear_vram():
    torch.cuda.empty_cache()
    gc.collect()

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
    print("加载 A-OKVQA 验证集（前 500 条样本）...")
    dataset = list(load_dataset("HuggingFaceM4/A-OKVQA", split="validation", streaming=True).take(500))
    
    qwen_id = "Qwen/Qwen2-VL-2B-Instruct"
    qwen_processor = AutoProcessor.from_pretrained(qwen_id)
    qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(qwen_id, torch_dtype=torch.float16, device_map=device)
    eval_baseline("Qwen2-VL-2B", qwen_model, qwen_processor, dataset, device)
    del qwen_model, qwen_processor
    clear_vram()
    
    llava_id = "llava-hf/llava-1.5-7b-hf"
    llava_processor = LlavaProcessor.from_pretrained(llava_id)
    llava_model = LlavaForConditionalGeneration.from_pretrained(llava_id, torch_dtype=torch.float16, device_map=device)
    eval_baseline("LLaVA-1.5-7B", llava_model, llava_processor, dataset, device)

if __name__ == "__main__":
    main()