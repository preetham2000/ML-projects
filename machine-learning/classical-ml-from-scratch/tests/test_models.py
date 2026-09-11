import unittest

import numpy as np

from models import (
    classification_accuracy,
    fit_lda,
    fit_ols,
    fit_qda,
    fit_ridge,
    mean_squared_error,
    polynomial_features,
    predict_lda,
    predict_linear,
    predict_qda,
    ridge_objective,
)


CLASSIFICATION_X = np.array(
    [[-3.0, -2.0], [-2.0, -3.0], [-3.0, -3.0], [-2.0, -2.0],
     [2.0, 2.0], [2.0, 3.0], [3.0, 2.0], [3.0, 3.0]]
)
CLASSIFICATION_Y = np.array([[10], [10], [10], [10], [20], [20], [20], [20]])
TEST_X = np.array([[-2.6, -2.4], [2.4, 2.6]])
TEST_Y = np.array([[10], [20]])

REGRESSION_X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
REGRESSION_Y = np.array([[1.0], [3.0], [5.0], [7.0]])


class DiscriminantAnalysisTests(unittest.TestCase):
    def test_lda_returns_original_labels_and_separates_accuracy(self):
        classes, means, covariance, priors = fit_lda(CLASSIFICATION_X, CLASSIFICATION_Y)
        predictions = predict_lda(TEST_X, classes, means, covariance, priors)

        np.testing.assert_array_equal(classes, [10, 20])
        np.testing.assert_allclose(means, [[-2.5, 2.5], [-2.5, 2.5]])
        np.testing.assert_allclose(covariance, np.diag([1 / 3, 1 / 3]))
        np.testing.assert_allclose(priors, [0.5, 0.5])
        np.testing.assert_array_equal(predictions, [10, 20])
        self.assertEqual(classification_accuracy(TEST_Y, predictions), 1.0)

    def test_qda_returns_original_labels_and_log_scores_remain_ordered(self):
        classes, means, covariances, priors = fit_qda(CLASSIFICATION_X, CLASSIFICATION_Y)
        predictions = predict_qda(TEST_X, classes, means, covariances, priors)
        distant_prediction = predict_qda(
            np.array([[1000.0, 1000.0]]), classes, means, covariances, priors
        )

        np.testing.assert_array_equal(classes, [10, 20])
        self.assertEqual(len(covariances), 2)
        for covariance in covariances:
            np.testing.assert_allclose(covariance, np.diag([1 / 3, 1 / 3]))
        np.testing.assert_array_equal(predictions, [10, 20])
        np.testing.assert_array_equal(distant_prediction, [20])
        self.assertEqual(classification_accuracy(TEST_Y.ravel(), predictions), 1.0)

    def test_classification_fits_flatten_targets_once(self):
        for target in (CLASSIFICATION_Y.ravel(), CLASSIFICATION_Y):
            with self.subTest(target_shape=target.shape):
                lda_model = fit_lda(CLASSIFICATION_X, target)
                qda_model = fit_qda(CLASSIFICATION_X, target)

                np.testing.assert_array_equal(lda_model[0], [10, 20])
                np.testing.assert_array_equal(qda_model[0], [10, 20])
                np.testing.assert_allclose(lda_model[3], [0.5, 0.5])
                np.testing.assert_allclose(qda_model[3], [0.5, 0.5])

    def test_supplied_priors_change_tied_predictions(self):
        lda_model = fit_lda(CLASSIFICATION_X, CLASSIFICATION_Y, priors=[0.1, 0.9])
        qda_model = fit_qda(CLASSIFICATION_X, CLASSIFICATION_Y, priors=[0.1, 0.9])
        ambiguous_point = np.array([[0.0, 0.0]])

        np.testing.assert_allclose(lda_model[3], [0.1, 0.9])
        np.testing.assert_allclose(qda_model[3], [0.1, 0.9])
        np.testing.assert_array_equal(predict_lda(ambiguous_point, *lda_model), [20])
        np.testing.assert_array_equal(predict_qda(ambiguous_point, *qda_model), [20])


class RegressionTests(unittest.TestCase):
    def test_ols_recovers_linear_predictions_and_near_zero_mse(self):
        weights = fit_ols(REGRESSION_X, REGRESSION_Y)
        predictions = predict_linear(REGRESSION_X, weights)

        np.testing.assert_allclose(weights, [1.0, 2.0])
        np.testing.assert_allclose(predictions, REGRESSION_Y.ravel())
        self.assertLess(mean_squared_error(REGRESSION_Y, predictions), 1e-20)

    def test_ols_handles_rank_deficient_design_matrix(self):
        X = np.column_stack([np.ones(4), np.arange(4), np.arange(4)])
        y = 1.0 + 2.0 * np.arange(4)

        weights = fit_ols(X, y)

        self.assertLess(mean_squared_error(y, predict_linear(X, weights)), 1e-20)

    def test_ridge_with_zero_penalty_matches_ols_for_supported_target_shapes(self):
        for target in (REGRESSION_Y.ravel(), REGRESSION_Y):
            with self.subTest(target_shape=target.shape):
                np.testing.assert_allclose(
                    fit_ridge(REGRESSION_X, target, 0), fit_ols(REGRESSION_X, target)
                )

    def test_ridge_with_zero_penalty_handles_rank_deficient_inputs(self):
        feature = np.arange(4, dtype=float)
        X = np.column_stack([np.ones(len(feature)), feature, feature])
        y = 1.0 + 2.0 * feature

        np.testing.assert_allclose(fit_ridge(X, y, 0), fit_ols(X, y))

    def test_ridge_leaves_a_leading_intercept_unpenalized_by_default(self):
        X = np.array([[1.0, 0.0], [1.0, 1.0]])
        y = np.array([1.0, 3.0])

        np.testing.assert_allclose(fit_ridge(X, y, 1.0), [5 / 3, 2 / 3])
        np.testing.assert_allclose(fit_ridge(X, y, 1.0, penalize_intercept=True), [1.0, 1.0])

    def test_ridge_objective_flattens_targets_and_matches_intercept_policy(self):
        weights = np.array([1.0, -1.0])
        X = np.array([[1.0, 0.0], [1.0, 1.0]])
        expected_gradient = np.array([-6.0, -7.0])

        for target in (np.array([1.0, 3.0]), np.array([[1.0], [3.0]])):
            with self.subTest(target_shape=target.shape):
                error, gradient = ridge_objective(weights, X, target, 0.5)

                self.assertEqual(error, 9.5)
                np.testing.assert_array_equal(gradient, expected_gradient)
                self.assertTrue(np.all(np.isfinite(gradient)))


class PolynomialFeatureTests(unittest.TestCase):
    def test_polynomial_features_accepts_flat_and_column_inputs(self):
        expected = np.array(
            [[1.0, -2.0, 4.0, -8.0], [1.0, 0.0, 0.0, 0.0], [1.0, 3.0, 9.0, 27.0]]
        )

        for x in (np.array([-2.0, 0.0, 3.0]), np.array([[-2.0], [0.0], [3.0]])):
            with self.subTest(input_shape=x.shape):
                np.testing.assert_array_equal(polynomial_features(x, 3), expected)


if __name__ == "__main__":
    unittest.main()
