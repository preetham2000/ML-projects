import unittest
from unittest.mock import patch

import numpy as np

from datasets import split_classification_data
from experiments import (
    _fit_standardizer,
    _standardize,
    classification_metrics,
    run_classification_experiment,
    run_regression_experiment,
    select_polynomial_parameters,
    select_ridge_lambda,
)


CLASSIFICATION_X = np.array(
    [[-4.0, -3.0], [-3.0, -4.0], [-4.0, -4.0], [-3.0, -3.0],
     [-4.0, -2.0], [-2.0, -4.0], [-2.0, -3.0], [-3.0, -2.0],
     [2.0, 2.0], [2.0, 3.0], [3.0, 2.0], [3.0, 3.0],
     [2.0, 4.0], [4.0, 2.0], [4.0, 3.0], [3.0, 4.0]]
)
CLASSIFICATION_Y = np.array([10] * 8 + [20] * 8)

REGRESSION_X = np.column_stack(
    [np.linspace(-3.0, 3.0, 60), np.cos(np.linspace(-3.0, 3.0, 60))]
)
REGRESSION_Y = (
    2.0
    + 1.5 * REGRESSION_X[:, 0]
    - 0.7 * REGRESSION_X[:, 1]
    + 0.2 * REGRESSION_X[:, 0] ** 2
)


class ClassificationExperimentTests(unittest.TestCase):
    def test_classification_metrics_include_accuracy_balance_and_confusion(self):
        metrics = classification_metrics([10, 10, 20, 20], [10, 20, 20, 20])

        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["balanced_accuracy"], 0.75)
        np.testing.assert_array_equal(metrics["labels"], [10, 20])
        np.testing.assert_array_equal(metrics["confusion_matrix"], [[1, 1], [0, 2]])

    def test_balanced_accuracy_ignores_predicted_only_labels(self):
        metrics = classification_metrics([10, 10], [10, 20])

        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)
        np.testing.assert_array_equal(metrics["labels"], [10, 20])
        np.testing.assert_array_equal(metrics["confusion_matrix"], [[1, 1], [0, 0]])

    def test_classification_workflow_is_reproducible_and_uses_a_holdout(self):
        first_result = run_classification_experiment(
            CLASSIFICATION_X, CLASSIFICATION_Y, test_size=0.25
        )
        second_result = run_classification_experiment(
            CLASSIFICATION_X, CLASSIFICATION_Y, test_size=0.25
        )

        self.assertEqual(first_result["n_train"], 12)
        self.assertEqual(first_result["n_test"], 4)
        for model_name in ("lda", "qda"):
            self.assertEqual(first_result[model_name]["accuracy"], 1.0)
            self.assertEqual(first_result[model_name]["balanced_accuracy"], 1.0)
            np.testing.assert_array_equal(
                first_result[model_name]["confusion_matrix"],
                second_result[model_name]["confusion_matrix"],
            )

    def test_classification_standardization_uses_training_statistics_only(self):
        X_train, _, _, _ = split_classification_data(CLASSIFICATION_X, CLASSIFICATION_Y, 0.25)

        with patch("experiments._fit_standardizer", wraps=_fit_standardizer) as standardizer:
            run_classification_experiment(CLASSIFICATION_X, CLASSIFICATION_Y, test_size=0.25)

        np.testing.assert_array_equal(standardizer.call_args.args[0], X_train)


class RegressionExperimentTests(unittest.TestCase):
    def test_standardization_uses_training_statistics_only(self):
        train = np.array([[0.0], [2.0]])
        test = np.array([[100.0]])
        mean, scale = _fit_standardizer(train)

        np.testing.assert_array_equal(_standardize(train, mean, scale), [[-1.0], [1.0]])
        np.testing.assert_array_equal(_standardize(test, mean, scale), [[99.0]])

    def test_cv_selectors_are_reproducible_and_only_accept_training_data(self):
        lambdas = (0.0, 0.1, 1.0)
        degrees = (1, 2)
        first_ridge = select_ridge_lambda(REGRESSION_X, REGRESSION_Y, lambdas, n_splits=4)
        second_ridge = select_ridge_lambda(REGRESSION_X, REGRESSION_Y, lambdas, n_splits=4)
        first_polynomial = select_polynomial_parameters(
            REGRESSION_X, REGRESSION_Y, 0, degrees, lambdas, n_splits=4
        )
        second_polynomial = select_polynomial_parameters(
            REGRESSION_X, REGRESSION_Y, 0, degrees, lambdas, n_splits=4
        )

        self.assertEqual(first_ridge, second_ridge)
        self.assertEqual(first_polynomial, second_polynomial)
        self.assertIn(first_ridge[0], lambdas)
        self.assertIn((first_polynomial[0], first_polynomial[1]), first_polynomial[2])

    def test_regression_workflow_selects_only_on_training_partition(self):
        lambdas = (0.0, 0.1, 1.0)
        degrees = (1, 2)

        with patch("experiments.select_ridge_lambda", wraps=select_ridge_lambda) as ridge_selector:
            with patch(
                "experiments.select_polynomial_parameters",
                wraps=select_polynomial_parameters,
            ) as polynomial_selector:
                result = run_regression_experiment(
                    REGRESSION_X,
                    REGRESSION_Y,
                    polynomial_feature_index=0,
                    lambdas=lambdas,
                    degrees=degrees,
                    test_size=0.2,
                    n_splits=4,
                )

        self.assertEqual(result["n_train"], 48)
        self.assertEqual(result["n_test"], 12)
        self.assertEqual(len(ridge_selector.call_args.args[0]), result["n_train"])
        self.assertEqual(len(polynomial_selector.call_args.args[0]), result["n_train"])
        self.assertIn(result["ridge"]["lambda"], lambdas)
        self.assertIn(result["polynomial"]["degree"], degrees)
        self.assertIn(result["polynomial"]["lambda"], lambdas)
        for model_name in ("ols", "ridge", "polynomial"):
            metrics = result[model_name] if model_name == "ols" else result[model_name]["test"]
            self.assertTrue(np.isfinite(metrics["mse"]))
            self.assertTrue(np.isfinite(metrics["rmse"]))
            self.assertTrue(np.isfinite(metrics["r2"]))


if __name__ == "__main__":
    unittest.main()
