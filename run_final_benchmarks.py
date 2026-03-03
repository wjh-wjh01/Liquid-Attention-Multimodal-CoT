import os
import torch
import gc
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from transformers import LlavaProcessor, LlavaForConditionalGeneration

# 使用镜像源下载数据集，避免网络不稳定
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def clear_vram():
    """清空显存，防止显存溢出"""
    torch.cuda.empty_cache()
    gc.collect()

def evaluate_model(model_name, model, processor, dataset, dataset_name, device):
    print(f"\n开始评测 {model_name} 在 {dataset_name} 上的表现...")
    correct = 0
    total = len(dataset)
    
    for i, item in tqdm(enumerate(dataset), total=total):
        try:
            question = item.get('question', '')
            choices = item.get('choices', [])
            answer_idx = item.get('answer', 0)
            
            # 兼容纯文本（MMLU-Pro）和图文（CMMCoT）
            image = item.get('image', None)
            if image is not None and not isinstance(image, str):
                image = image.convert('RGB')
                
            choices_str = " ".join([f"({chr(65+j)}) {c}" for j, c in enumerate(choices)])
            correct_answer = str(choices[answer_idx] if isinstance(answer_idx, int) else answer_idx).lower()
            
            prompt = f"Question: {question}\nChoices: {choices_str}\nAnswer with the correct choice directly."
            
            # 不同模型的预处理逻辑
            if "qwen" in model_name.lower():
                messages = [{"role": "user", "content": []}]
                if image:
                    messages[0]["content"].append({"type": "image", "image": image})
                messages[0]["content"].append({"type": "text", "text": prompt})
                
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=[text],
                    images=[image] if image else None,
                    padding=True,
                    return_tensors="pt"
                ).to(device)
            else:  # LLaVA
                if image:
                    prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
                    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float16)
                else:
                    prompt = f"USER: {prompt}\nASSISTANT:"
                    inputs = processor(text=prompt, return_tensors="pt").to(device, torch.float16)

            input_len = inputs["input_ids"].shape[1]
            
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=False
                )
                
            generated_ids = output_ids[0][input_len:]
            pred_text = processor.decode(generated_ids, skip_special_tokens=True).lower().strip()
            
            if correct_answer in pred_text or chr(65+answer_idx).lower() in pred_text:
                correct += 1
        except Exception:
            continue  # 跳过异常样本，避免中断评测
            
    acc = correct / total if total > 0 else 0
    print(f"{model_name} 在 {dataset_name} 上的准确率: {acc*100:.2f}%")
    return acc

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 1. 加载数据集（各抽取 500 条样本进行零样本评测）
    print("\n加载数据集...")
    try:
        mmlu_pro = load_dataset("TIGER-Lab/MMLU-Pro", split="test", streaming=True).take(500)
        mmlu_pro_data = list(mmlu_pro)
    except:
        print("MMLU-Pro 加载失败，将跳过该数据集。")
        mmlu_pro_data = []

    try:
        cmmcot = load_dataset("CMMCoT-dataset-path", split="test", streaming=True).take(500)
        cmmcot_data = list(cmmcot)
    except:
        print("CMMCoT 加载失败，将跳过该数据集。")
        cmmcot_data = []

    # 2. 评测 Qwen2-VL-2B
    qwen_id = "Qwen/Qwen2-VL-2B-Instruct"
    print(f"\n加载模型: {qwen_id}")
    qwen_processor = AutoProcessor.from_pretrained(qwen_id)
    qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
        qwen_id,
        torch_dtype=torch.float16,
        device_map=device
    )
    qwen_model.eval()
    
    if mmlu_pro_data:
        evaluate_model("Qwen2-VL-2B", qwen_model, qwen_processor, mmlu_pro_data, "MMLU-Pro", device)
    if cmmcot_data:
        evaluate_model("Qwen2-VL-2B", qwen_model, qwen_processor, cmmcot_data, "CMMCoT", device)
    
    del qwen_model, qwen_processor
    clear_vram()

    # 3. 评测 LLaVA-1.5-7B
    llava_id = "llava-hf/llava-1.5-7b-hf"
    print(f"\n加载模型: {llava_id}")
    llava_processor = LlavaProcessor.from_pretrained(llava_id)
    llava_model = LlavaForConditionalGeneration.from_pretrained(
        llava_id,
        torch_dtype=torch.float16,
        device_map=device
    )
    llava_model.eval()
    
    if mmlu_pro_data:
        evaluate_model("LLaVA-1.5", llava_model, llava_processor, mmlu_pro_data, "MMLU-Pro", device)
    if cmmcot_data:
        evaluate_model("LLaVA-1.5", llava_model, llava_processor, cmmcot_data, "CMMCoT", device)

    print("\n基线测试完成。")

if __name__ == "__main__":
    main()