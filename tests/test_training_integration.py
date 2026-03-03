import json
import tempfile
import unittest
from pathlib import Path

from src.training.pipeline import run_single_experiment


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


class TestTrainingIntegration(unittest.TestCase):
    def test_one_epoch_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ds_dir = root / "data" / "processed" / "scienceqa"

            train = []
            for i in range(40):
                train.append(
                    {
                        "id": f"tr-{i}",
                        "dataset": "scienceqa",
                        "split": "train",
                        "question": f"What is {i}+{i}?",
                        "choices": [str(i), str(2 * i), str(3 * i), "0"],
                        "answer_idx": 1,
                        "image_path": None,
                        "difficulty": "simple" if i % 2 == 0 else "difficulty",
                        "metadata": {},
                    }
                )
            val = train[:10]
            test = train[10:20]

            _write_jsonl(ds_dir / "train.jsonl", train)
            _write_jsonl(ds_dir / "val.jsonl", val)
            _write_jsonl(ds_dir / "test.jsonl", test)

            cfg = {
                "seed": 3407,
                "output_dir": str(root / "outputs" / "runs"),
                "dataset": {
                    "name": "scienceqa",
                    "processed_root": str(root / "data" / "processed"),
                },
                "model": {
                    "hidden_dim": 32,
                    "image_dim": 16,
                    "vocab_size": 2000,
                    "max_question_len": 32,
                    "max_choice_len": 16,
                    "max_reasoning_steps": 3,
                    "min_reasoning_steps": 1,
                    "stop_threshold": 0.8,
                    "attention_mode": "liquid",
                    "tau": 0.5,
                    "dt": 0.2,
                    "micro_steps": 2,
                    "dropout": 0.1,
                },
                "train": {
                    "epochs": 1,
                    "batch_size": 8,
                    "lr": 0.001,
                    "weight_decay": 0.0001,
                    "grad_clip": 1.0,
                    "log_every": 200,
                },
                "eval": {
                    "batch_size": 8,
                    "split": "test",
                    "noise_levels": [0.1, 0.2],
                },
                "ablation_flags": {
                    "use_ode": True,
                    "use_cross_step": True,
                    "use_self_validation": True,
                    "use_multimodal": True,
                    "use_cot_control": True,
                    "use_reflection": False,
                },
            }

            result = run_single_experiment(
                run_cfg=cfg,
                experiment_name="liquid_full",
                dataset_name="scienceqa",
            )

            self.assertIn("test_accuracy", result)
            self.assertTrue((Path(result["run_dir"]) / "checkpoint.pt").exists())


if __name__ == "__main__":
    unittest.main()
