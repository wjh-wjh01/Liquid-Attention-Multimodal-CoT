import os
from datasets import load_dataset

print("开始下载完整 ScienceQA 数据集...")

# 设置 HuggingFace 国内镜像加速
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 下载完整数据集 (Train, Val, Test)
# 注意：可能会占用较多硬盘空间和带宽
try:
    ds = load_dataset("derek-thomas/ScienceQA")
    
    # 确保保存目录存在
    save_dir = "/root/autodl-tmp/project/data/raw/scienceqa_full/data"
    os.makedirs(save_dir, exist_ok=True)
    
    print("数据集下载成功，正在转换为 Parquet 格式并保存...")
    
    # 保存为 parquet 格式
    ds['train'].to_parquet(os.path.join(save_dir, "train-full.parquet"))
    ds['validation'].to_parquet(os.path.join(save_dir, "val-full.parquet"))
    ds['test'].to_parquet(os.path.join(save_dir, "test-full.parquet"))
    
    print(f"保存完成，路径：{save_dir}")
except Exception as e:
    print(f"下载失败: {e}")