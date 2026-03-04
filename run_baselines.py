import os
import io
import gc
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import LlavaProcessor, LlavaForConditionalGeneration

# 禁用 flash-attn CUDA 构建
os.environ["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "TRUE"


def clear_vram():
    """清空显存，防止 OOM"""
    gc.collect()
    torch.cuda.empty_cache()


def evaluate_llava(model_id, model_name, df_test, device="cuda"):
    print(f"\n{'='*50}")
    print(f"Loading {model_name} ({model_id})...")

    processor = LlavaProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device
    )

    model.eval()
    print(f"{model_name} loaded. Running evaluation...")

    correct = 0
    total = len(df_test)

    for _, row in tqdm(df_test.iterrows(), total=total):
        question = row['question']
        choices = row['choices']
        answer_idx = row['answer']

        # 处理图片
        img_data = row['image']
        if isinstance(img_data, dict) and 'bytes' in img_data:
            image = Image.open(io.BytesIO(img_data['bytes'])).convert('RGB')
        else:
            image = img_data.convert('RGB')

        choices_str = " ".join(
            [f"({chr(97 + j)}) {c}" for j, c in enumerate(choices)]
        )

        prompt = (
            "USER: <image>\n"
            f"Question: {question}\n"
            f"Choices: {choices_str}\n"
            "Answer with the correct choice directly.\n"
            "ASSISTANT:"
        )

        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(device, torch.float16)

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

        correct_choice_text = str(choices[answer_idx]).lower()

        if (
            correct_choice_text in pred_text
            or pred_text in correct_choice_text
        ):
            correct += 1

    acc = correct / total
    print(f"{model_name} evaluation finished. Accuracy: {acc*100:.2f}%")

    del model
    del processor
    clear_vram()

    return acc


def main():
    parquet_path = "/root/autodl-tmp/project/data/raw/scienceqa_full/data/test-full.parquet"
    print(f"Loading test set: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    df_img = (
        df[df['image'].notnull()]
        .sample(500, random_state=42)
        .reset_index(drop=True)
    )

    print("Sampling 500 multimodal questions for evaluation...")

    acc_llava = evaluate_llava(
        "llava-hf/llava-1.5-7b-hf",
        "LLaVA-1.5 (7B)",
        df_img
    )

    print("\n" + "="*50)
    print("Baseline result")
    print(f"LLaVA-1.5 : {acc_llava*100:.2f}%")
    print("="*50)


if __name__ == "__main__":
    main()
