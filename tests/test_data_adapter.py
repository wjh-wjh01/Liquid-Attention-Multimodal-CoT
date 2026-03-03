import json
import tempfile
import unittest
from pathlib import Path

from src.data.adapters import prepare_dataset


class TestDataAdapter(unittest.TestCase):
    def test_prepare_dataset_schema(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "raw" / "scienceqa"
            processed = root / "processed"
            raw.mkdir(parents=True, exist_ok=True)

            rows = [
                {
                    "question": "What is 2+2?",
                    "choices": ["1", "2", "4", "8"],
                    "answer": "C",
                    "split": "train",
                },
                {
                    "question": "Sky color?",
                    "choices": ["blue", "green"],
                    "answer_idx": 0,
                    "split": "val",
                },
                {
                    "question": "Fire is?",
                    "choices": ["cold", "hot"],
                    "answer": "hot",
                    "split": "test",
                },
            ]
            with (raw / "data.jsonl").open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")

            report = prepare_dataset(
                dataset_name="scienceqa",
                raw_root=root / "raw",
                processed_root=processed,
                subset_sizes={"train": 10, "val": 10, "test": 10},
                seed=3407,
            )

            self.assertEqual(report["status"], "ok")
            train_path = processed / "scienceqa" / "train.jsonl"
            self.assertTrue(train_path.exists())
            line = train_path.read_text(encoding="utf-8").splitlines()[0]
            obj = json.loads(line)
            for key in [
                "id",
                "dataset",
                "split",
                "question",
                "choices",
                "answer_idx",
                "image_path",
                "difficulty",
                "metadata",
            ]:
                self.assertIn(key, obj)


if __name__ == "__main__":
    unittest.main()
