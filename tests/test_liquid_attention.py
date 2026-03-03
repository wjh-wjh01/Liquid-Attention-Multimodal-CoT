import unittest

import torch

from src.models.liquid_attention import LiquidAttention


class TestLiquidAttention(unittest.TestCase):
    def test_row_normalized_and_finite(self) -> None:
        torch.manual_seed(0)
        module = LiquidAttention(hidden_dim=16, tau=0.5, dt=0.2, micro_steps=3)
        token_states = torch.randn(2, 12, 16)
        reasoning_state = torch.randn(2, 16)

        attn, entropy = module(token_states, reasoning_state, mode="liquid", use_ode=True, use_cross_step=True)
        self.assertEqual(attn.shape, (2, 12, 12))
        self.assertEqual(entropy.shape, (2,))

        row_sums = attn.sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4))
        self.assertTrue(torch.isfinite(attn).all())
        self.assertTrue(torch.isfinite(entropy).all())


if __name__ == "__main__":
    unittest.main()
