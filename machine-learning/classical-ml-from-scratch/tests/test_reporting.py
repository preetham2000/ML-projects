import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from run_experiments import generate_outputs


WINE_X = np.array(
    [[-4.0, -3.0], [-3.0, -4.0], [-4.0, -4.0], [-3.0, -3.0],
     [-4.0, -2.0], [-2.0, -4.0], [-2.0, -3.0], [-3.0, -2.0],
     [2.0, 2.0], [2.0, 3.0], [3.0, 2.0], [3.0, 3.0],
     [2.0, 4.0], [4.0, 2.0], [4.0, 3.0], [3.0, 4.0]]
)
WINE_Y = np.array([10] * 8 + [20] * 8)
RNG = np.random.default_rng(11)
DIABETES_X = RNG.normal(size=(60, 10))
DIABETES_Y = 2.0 + 1.2 * DIABETES_X[:, 0] - 0.5 * DIABETES_X[:, 1] + 0.3 * DIABETES_X[:, 2] ** 2


class ReportingTests(unittest.TestCase):
    def test_runner_generates_figures_and_machine_readable_results(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory)
            with patch("run_experiments.load_wine_data", return_value=(WINE_X, WINE_Y)):
                with patch("run_experiments.load_diabetes_data", return_value=(DIABETES_X, DIABETES_Y)):
                    results = generate_outputs(output_path)

            expected_files = (
                "classification_wine_summary.png",
                "regression_diabetes_summary.png",
                "results.json",
            )
            for filename in expected_files:
                self.assertTrue((output_path / filename).is_file())
                self.assertGreater((output_path / filename).stat().st_size, 0)

            saved_results = json.loads((output_path / "results.json").read_text())
            self.assertEqual(saved_results["random_seed"], 42)
            self.assertEqual(saved_results["classification"]["n_test"], results["classification"]["n_test"])
            self.assertIn("cv_mse", saved_results["regression"]["ridge"])
            self.assertIn("degree", saved_results["regression"]["polynomial"])
            polynomial_scores = saved_results["regression"]["polynomial"]["cv_mse"]
            self.assertIsInstance(polynomial_scores, list)
            self.assertTrue(polynomial_scores)
            self.assertEqual(set(polynomial_scores[0]), {"degree", "lambda", "mse"})


if __name__ == "__main__":
    unittest.main()
