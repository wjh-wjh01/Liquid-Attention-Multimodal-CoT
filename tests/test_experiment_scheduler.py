import unittest

from src.common.config import load_yaml
from src.common.experiments import parse_experiment_matrix


class TestExperimentScheduler(unittest.TestCase):
    def test_matrix_layout(self) -> None:
        cfg = load_yaml("configs/experiments/experiment_matrix.yaml")
        matrix = parse_experiment_matrix(cfg)

        self.assertEqual(len(matrix.datasets), 3)
        self.assertEqual(len(matrix.experiments), 9)
        self.assertIn("w_o_ode", matrix.aliases)
        self.assertEqual(matrix.aliases["w_o_ode"], "static_cot")


if __name__ == "__main__":
    unittest.main()
