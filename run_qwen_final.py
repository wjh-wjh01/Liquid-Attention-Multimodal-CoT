import os
import io
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def main():
    print("启动 Qwen2-VL 基线测试 (2B)")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parquet_path = "/root/autodl-tmp/project/data/raw/scienceqa_full/data/test-full.parquet"
    df = pd.read_parquet(parquet_path)
    df_img = df[df['image'].notnull()].sample(500, random_state=42).reset_index(drop=True)
    
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    print(f"\n加载模型: {model_id}")
    
    # 使用 PyTorch SDPA 注意力实现
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device
    )
    model.eval()
    
    correct = 0
    total = len(df_img)
    
    for i, row in tqdm(df_img.iterrows(), total=total):
        question = row['question']
        choices = row['choices']
        answer_idx = row['answer']
        
        img_data = row['image']
        if isinstance(img_data, dict) and 'bytes' in img_data:
            image = Image.open(io.BytesIO(img_data['bytes'])).convert('RGB')
        else:
            image = img_data.convert('RGB')
            
        choices_str = " ".join([f"({chr(97+j)}) {c}" for j, c in enumerate(choices)])
        correct_answer = str(choices[answer_idx]).lower()
        
        # Qwen2-VL 官方对话格式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Question: {question}\nChoices: {choices_str}\nAnswer with the correct choice directly."},
                ],
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(device)

        input_len = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=15,
                temperature=0.1,
                do_sample=False
            )
            
        generated_ids = output_ids[0][input_len:]
        pred_text = processor.decode(
            generated_ids,
            skip_special_tokens=True
        ).lower().strip()
        
        if correct_answer in pred_text:
            correct += 1
            
    acc = correct / total

    print("\n" + "="*50)
    print("Baseline evaluation summary")
    print(f"LLaVA-1.5 (2023)  : 48.40%")
    print(f"Qwen2-VL-2B (2024) : {acc*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()