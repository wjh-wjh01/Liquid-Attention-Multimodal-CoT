import yaml
import torch
from torch.utils.data import DataLoader

from src.models.multimodal_cot import MultimodalCoTModel
from src.data.dataset import DatasetConfig, JsonlReasoningDataset
from src.data.tokenizer import SimpleTokenizer


RUN_DIR = "/root/autodl-tmp/project/code/outputs/runs/dummy_scienceqa/liquid_real_train/seed_3407"
TEST_JSONL = "/root/autodl-tmp/project/data/processed/dummy_scienceqa/test.jsonl"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(f"{RUN_DIR}/config_snapshot.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    tokenizer = SimpleTokenizer.load(f"{RUN_DIR}/tokenizer.json")

    ds_cfg = DatasetConfig(
        max_question_len=int(cfg["model"]["max_question_len"]),
        max_choice_len=int(cfg["model"]["max_choice_len"]),
        image_dim=int(cfg["model"]["image_dim"]),
        noise_prob=0.0
    )
    
    ds = JsonlReasoningDataset(
        jsonl_path=TEST_JSONL,
        tokenizer=tokenizer,
        cfg=ds_cfg,
        include_unlabeled=False,
        seed=3407
    )

    test_loader = DataLoader(
        ds,
        batch_size=32,
        shuffle=False,
        collate_fn=ds.collate_fn
    )

    model = MultimodalCoTModel(
        model_cfg=cfg["model"],
        default_flags=cfg["ablation_flags"]
    ).to(device)

    ckpt = torch.load(f"{RUN_DIR}/checkpoint.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    correct, total = 0, 0

    with torch.no_grad():
        for batch in test_loader:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            out = model(batch, flags=cfg["ablation_flags"])

            valid = batch["answer_idx"] >= 0
            if valid.sum() > 0:
                preds = torch.argmax(out["logits"][valid], dim=-1)
                correct += (preds == batch["answer_idx"][valid]).sum().item()
                total += valid.sum().item()

    print("\n" + "=" * 50)
    print("Neural ODE (Liquid Attention) Evaluation Finished")
    print("=" * 50)
    print(f"Test Accuracy: {correct / max(total, 1) * 100:.2f}%")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
