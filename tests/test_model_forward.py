import unittest

import torch

from src.models.multimodal_cot import MultimodalCoTModel


class TestModelForward(unittest.TestCase):
    def test_forward_contract(self) -> None:
        model_cfg = {
            "hidden_dim": 32,
            "image_dim": 16,
            "vocab_size": 500,
            "max_question_len": 20,
            "max_choice_len": 10,
            "max_reasoning_steps": 4,
            "min_reasoning_steps": 1,
            "stop_threshold": 0.7,
            "attention_mode": "liquid",
            "tau": 0.5,
            "dt": 0.2,
            "micro_steps": 2,
            "dropout": 0.1,
        }
        flags = {
            "use_ode": True,
            "use_cross_step": True,
            "use_self_validation": True,
            "use_multimodal": True,
            "use_cot_control": True,
            "use_reflection": False,
        }

        model = MultimodalCoTModel(model_cfg, flags)
        batch = {
            "question_ids": torch.randint(0, 200, (3, 20)),
            "question_mask": torch.ones(3, 20, dtype=torch.bool),
            "choice_ids": torch.randint(0, 200, (3, 4, 10)),
            "choice_mask": torch.ones(3, 4, 10, dtype=torch.bool),
            "choice_valid": torch.ones(3, 4, dtype=torch.bool),
            "answer_idx": torch.tensor([0, 1, 2]),
            "image_feats": torch.randn(3, 16),
        }

        out = model(batch)
        self.assertIn("logits", out)
        self.assertIn("stop_prob", out)
        self.assertIn("steps_used", out)
        self.assertIn("attn_state", out)
        self.assertIn("attn_entropy", out)

        self.assertEqual(out["logits"].shape, (3, 4))
        self.assertEqual(out["stop_prob"].shape, (3,))
        self.assertEqual(out["steps_used"].shape, (3,))


if __name__ == "__main__":
    unittest.main()
