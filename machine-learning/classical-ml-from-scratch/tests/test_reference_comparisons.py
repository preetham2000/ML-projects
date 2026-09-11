import unittest

import numpy as np
from scipy.optimize import minimize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures

from models import (
    fit_lda,
    fit_ols,
    fit_qda,
    fit_ridge,
    polynomial_features,
    predict_lda,
    predict_linear,
    predict_qda,
    ridge_objective,
)


RNG = np.random.default_rng(7)
CLASS_LABELS = np.array([10, 20, 40])
CLASS_PRIORS = np.repeat(1 / 3, 3)
CLASS_COVARIANCE = np.array([[1.0, 0.25], [0.25, 0.8]])
CLASS_CENTERS = np.array([[-3.0, -2.0], [0.0, 3.0], [3.0, -1.0]])
CLASSIFICATION_X = np.vstack(
    [RNG.multivariate_normal(center, CLASS_COVARIANCE, size=12) for center in CLASS_CENTERS]
)
CLASSIFICATION_Y = np.repeat(CLASS_LABELS, 12)
IMBALANCED_X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
IMBALANCED_Y = np.array([10, 10, 10, 20, 20, 20, 20, 20, 20])
BOUNDARY_POINTS = np.array([[0.5], [0.85], [1.1]])

REGRESSION_FEATURES = RNG.normal(size=(30, 2))
REGRESSION_X = np.column_stack([np.ones(len(REGRESSION_FEATURES)), REGRESSION_FEATURES])
REGRESSION_Y = 1.0 + 2.0 * REGRESSION_FEATURES[:, 0] - 0.5 * REGRESSION_FEATURES[:, 1]


def finite_difference_gradient(
    weights: np.ndarray, X: np.ndarray, y: np.ndarray, lambda_: float, epsilon: float = 1e-6
) -> np.ndarray:
    """Approximate the ridge gradient with centered finite differences."""
    gradient = np.empty_like(weights, dtype=float)
    for index in range(len(weights)):
        offset = np.zeros_like(weights, dtype=float)
        offset[index] = epsilon
        loss_plus, _ = ridge_objective(weights + offset, X, y, lambda_)
        loss_minus, _ = ridge_objective(weights - offset, X, y, lambda_)
        gradient[index] = (loss_plus - loss_minus) / (2 * epsilon)
    return gradient


class DiscriminantReferenceTests(unittest.TestCase):
    def test_lda_matches_sklearn_on_a_balanced_equal_prior_fixture(self):
        custom_model = fit_lda(CLASSIFICATION_X, CLASSIFICATION_Y, CLASS_PRIORS)
        sklearn_model = LinearDiscriminantAnalysis(solver="lsqr", priors=CLASS_PRIORS).fit(
            CLASSIFICATION_X, CLASSIFICATION_Y
        )
        classes, means, covariance, priors = custom_model

        scale_factor = len(CLASSIFICATION_X) / (len(CLASSIFICATION_X) - len(classes))
        custom_predictions = predict_lda(CLASSIFICATION_X, *custom_model)
        sklearn_predictions = sklearn_model.predict(CLASSIFICATION_X)

        np.testing.assert_array_equal(classes, CLASS_LABELS)
        np.testing.assert_allclose(priors, sklearn_model.priors_, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(means.T, sklearn_model.means_, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            covariance, sklearn_model.covariance_ * scale_factor, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_array_equal(custom_predictions, sklearn_predictions)

    def test_qda_matches_sklearn_on_a_balanced_equal_prior_fixture(self):
        custom_model = fit_qda(CLASSIFICATION_X, CLASSIFICATION_Y, CLASS_PRIORS)
        sklearn_model = QuadraticDiscriminantAnalysis(
            priors=CLASS_PRIORS, store_covariance=True
        ).fit(CLASSIFICATION_X, CLASSIFICATION_Y)
        classes, means, covariances, priors = custom_model

        custom_predictions = predict_qda(CLASSIFICATION_X, *custom_model)
        sklearn_predictions = sklearn_model.predict(CLASSIFICATION_X)

        np.testing.assert_array_equal(classes, CLASS_LABELS)
        np.testing.assert_allclose(priors, sklearn_model.priors_, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(means.T, sklearn_model.means_, rtol=1e-12, atol=1e-12)
        for covariance, sklearn_covariance in zip(covariances, sklearn_model.covariance_):
            scale_factor = 12 / 11
            np.testing.assert_allclose(
                covariance, sklearn_covariance * scale_factor, rtol=1e-12, atol=1e-12
            )
        np.testing.assert_array_equal(custom_predictions, sklearn_predictions)

    def test_imbalanced_reference_comparison_preserves_covariance_convention(self):
        lda_model = fit_lda(IMBALANCED_X, IMBALANCED_Y)
        qda_model = fit_qda(IMBALANCED_X, IMBALANCED_Y)
        sklearn_lda = LinearDiscriminantAnalysis(solver="lsqr").fit(IMBALANCED_X, IMBALANCED_Y)
        sklearn_qda = QuadraticDiscriminantAnalysis(store_covariance=True).fit(
            IMBALANCED_X, IMBALANCED_Y
        )
        classes, means, covariance, priors = lda_model
        qda_classes, qda_means, covariances, qda_priors = qda_model
        _, counts = np.unique(IMBALANCED_Y, return_counts=True)

        np.testing.assert_array_equal(classes, sklearn_lda.classes_)
        np.testing.assert_array_equal(qda_classes, sklearn_qda.classes_)
        np.testing.assert_allclose(means.T, sklearn_lda.means_, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(qda_means.T, sklearn_qda.means_, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(priors, sklearn_lda.priors_, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(qda_priors, sklearn_qda.priors_, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            covariance,
            sklearn_lda.covariance_ * len(IMBALANCED_X) / (len(IMBALANCED_X) - len(classes)),
            rtol=1e-12,
            atol=1e-12,
        )
        for covariance, sklearn_covariance, count in zip(covariances, sklearn_qda.covariance_, counts):
            np.testing.assert_allclose(
                covariance, sklearn_covariance * count / (count - 1), rtol=1e-12, atol=1e-12
            )

        lda_predictions = predict_lda(BOUNDARY_POINTS, *lda_model)
        qda_predictions = predict_qda(BOUNDARY_POINTS, *qda_model)
        sklearn_lda_predictions = sklearn_lda.predict(BOUNDARY_POINTS)
        sklearn_qda_predictions = sklearn_qda.predict(BOUNDARY_POINTS)
        self.assertTrue(np.any(lda_predictions != sklearn_lda_predictions))
        self.assertTrue(np.any(qda_predictions != sklearn_qda_predictions))

    def test_custom_qda_reports_singular_covariance(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [5.0, 5.0], [6.0, 6.0], [7.0, 7.0]])
        y = np.array([10, 10, 10, 40, 40, 40])
        model = fit_qda(X, y)

        with self.assertRaises(np.linalg.LinAlgError):
            predict_qda(X, *model)


class RegressionReferenceTests(unittest.TestCase):
    def test_ols_matches_sklearn_for_flat_and_column_targets(self):
        sklearn_model = LinearRegression(fit_intercept=False).fit(REGRESSION_X, REGRESSION_Y)

        for target in (REGRESSION_Y, REGRESSION_Y.reshape(-1, 1)):
            with self.subTest(target_shape=target.shape):
                weights = fit_ols(REGRESSION_X, target)
                predictions = predict_linear(REGRESSION_X, weights)

                np.testing.assert_allclose(weights, sklearn_model.coef_, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(
                    predictions, sklearn_model.predict(REGRESSION_X), rtol=1e-12, atol=1e-12
                )

    def test_ols_matches_sklearn_predictions_on_rank_deficient_data(self):
        feature = np.arange(6, dtype=float)
        X = np.column_stack([np.ones(len(feature)), feature, feature])
        y = 1.0 + 2.0 * feature
        custom_predictions = predict_linear(X, fit_ols(X, y))
        sklearn_predictions = LinearRegression(fit_intercept=False).fit(X, y).predict(X)

        np.testing.assert_allclose(custom_predictions, sklearn_predictions, rtol=1e-12, atol=1e-12)

    def test_ridge_matches_sklearn_with_an_unpenalized_intercept(self):
        lambda_ = 0.7
        sklearn_model = Ridge(alpha=lambda_, fit_intercept=True).fit(
            REGRESSION_FEATURES, REGRESSION_Y
        )

        for target in (REGRESSION_Y, REGRESSION_Y.reshape(-1, 1)):
            with self.subTest(target_shape=target.shape):
                weights = fit_ridge(REGRESSION_X, target, lambda_)
                expected_weights = np.concatenate([[sklearn_model.intercept_], sklearn_model.coef_])

                np.testing.assert_allclose(weights, expected_weights, rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(
                    predict_linear(REGRESSION_X, weights),
                    sklearn_model.predict(REGRESSION_FEATURES),
                    rtol=1e-10,
                    atol=1e-12,
                )

    def test_polynomial_features_match_sklearn_with_a_bias_column(self):
        transformer = PolynomialFeatures(degree=4, include_bias=True)
        expected = transformer.fit_transform(REGRESSION_FEATURES[:, :1])

        for x in (REGRESSION_FEATURES[:, 0], REGRESSION_FEATURES[:, :1]):
            with self.subTest(input_shape=x.shape):
                np.testing.assert_allclose(
                    polynomial_features(x, 4), expected, rtol=1e-12, atol=1e-12
                )

    def test_analytic_ridge_gradient_matches_centered_finite_difference(self):
        weights = np.array([0.3, -0.2, 0.5])
        analytic_gradient = ridge_objective(weights, REGRESSION_X, REGRESSION_Y, 0.7)[1]
        numerical_gradient = finite_difference_gradient(weights, REGRESSION_X, REGRESSION_Y, 0.7)

        np.testing.assert_allclose(analytic_gradient, numerical_gradient, rtol=1e-5, atol=1e-6)

    def test_closed_form_ridge_matches_scipy_minimize(self):
        lambda_ = 0.7
        closed_form_weights = fit_ridge(REGRESSION_X, REGRESSION_Y, lambda_)
        result = minimize(
            ridge_objective,
            x0=np.zeros(REGRESSION_X.shape[1]),
            args=(REGRESSION_X, REGRESSION_Y, lambda_),
            jac=True,
            method="BFGS",
            options={"gtol": 1e-10, "maxiter": 500},
        )

        self.assertTrue(result.success, result.message)
        np.testing.assert_allclose(result.x, closed_form_weights, rtol=1e-8, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
