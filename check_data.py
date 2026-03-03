import glob
from datasets import load_dataset

print("📡 启动全局雷达，扫描 600M Parquet 金矿...")


train_files = glob.glob("/root/autodl-tmp/project/**/*train*.parquet", recursive=True)
val_files = glob.glob("/root/autodl-tmp/project/**/*validation*.parquet", recursive=True)
test_files = glob.glob("/root/autodl-tmp/project/**/*test*.parquet", recursive=True)

data_files = {}
if train_files: data_files["train"] = train_files
if val_files: data_files["validation"] = val_files
if test_files: data_files["test"] = test_files

print(f" 找到金矿坐标: {data_files}")

if not data_files:
    print("Parquet 文件消失")
else:
    print(" (load_dataset)...")
    # 直接用绝对路径强行加载，无视任何相对目录的报错！
    ds = load_dataset("parquet", data_files=data_files)
    print("真实的 ScienceQA 数据结构如下：")
    print(ds)