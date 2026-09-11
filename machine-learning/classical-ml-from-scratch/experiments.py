"""Reproducible evaluation workflows for the from-scratch models."""

import numpy as np

from datasets import (
    RANDOM_SEED,
    load_diabetes_data,
    load_wine_data,
    split_classification_data,
    split_regression_data,
)
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
)


DEFAULT_LAMBDAS = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0)
DEFAULT_DEGREES = (1, 2, 3, 4)


def classification_metrics(
    y_true: np.ndarray, predictions: np.ndarray
) -> dict[str, np.ndarray | float]:
    """Return accuracy, balanced accuracy, labels, and a confusion matrix."""
    y_true = np.asarray(y_true).ravel()
    predictions = np.asarray(predictions).ravel()
    labels = np.unique(np.concatenate([y_true, predictions]))
    confusion = np.zeros((len(labels), len(labels)), dtype=int)
    for row, label in enumerate(labels):
        for column, predicted_label in enumerate(labels):
            confusion[row, column] = np.sum((y_true == label) & (predictions == predicted_label))

    supports = confusion.sum(axis=1)
    recalls = np.diag(confusion)[supports > 0] / supports[supports > 0]
    return {
        "accuracy": classification_accuracy(y_true, predictions),
        "balanced_accuracy": float(np.mean(recalls)),
        "labels": labels,
        "confusion_matrix": confusion,
    }


def regression_metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Return MSE, RMSE, and R-squared for regression predictions."""
    y_true = np.asarray(y_true).ravel()
    predictions = np.asarray(predictions).ravel()
    mse = mean_squared_error(y_true, predictions)
    total_sum_squares = np.sum((y_true - y_true.mean()) ** 2)
    residual_sum_squares = np.sum((y_true - predictions) ** 2)
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(1 - residual_sum_squares / total_sum_squares),
    }


def _fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return feature means and scales fitted on training data."""
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    return mean, scale


def _standardize(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Apply training-derived feature scaling."""
    return (np.asarray(X, dtype=float) - mean) / scale


def _add_intercept(X: np.ndarray) -> np.ndarray:
    """Add a leading intercept column."""
    return np.column_stack([np.ones(len(X)), X])


def _fold_indices(n_samples: int, n_splits: int, random_state: int):
    """Yield reproducible shuffled train/validation index pairs."""
    indices = np.random.default_rng(random_state).permutation(n_samples)
    for validation_indices in np.array_split(indices, n_splits):
        train_indices = np.setdiff1d(indices, validation_indices, assume_unique=True)
        yield train_indices, validation_indices


def select_ridge_lambda(
    X: np.ndarray,
    y: np.ndarray,
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS,
    n_splits: int = 5,
    random_state: int = RANDOM_SEED,
) -> tuple[float, dict[float, float]]:
    """Select a ridge penalty from training-only cross-validation MSE."""
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    scores = {}
    for lambda_ in lambdas:
        fold_mse = []
        for train_indices, validation_indices in _fold_indices(len(y), n_splits, random_state):
            mean, scale = _fit_standardizer(X[train_indices])
            X_train = _add_intercept(_standardize(X[train_indices], mean, scale))
            X_validation = _add_intercept(_standardize(X[validation_indices], mean, scale))
            weights = fit_ridge(X_train, y[train_indices], lambda_)
            predictions = predict_linear(X_validation, weights)
            fold_mse.append(mean_squared_error(y[validation_indices], predictions))
        scores[float(lambda_)] = float(np.mean(fold_mse))

    best_lambda = min(scores, key=scores.get)
    return best_lambda, scores


def select_polynomial_parameters(
    X: np.ndarray,
    y: np.ndarray,
    feature_index: int,
    degrees: tuple[int, ...] = DEFAULT_DEGREES,
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS,
    n_splits: int = 5,
    random_state: int = RANDOM_SEED,
) -> tuple[int, float, dict[tuple[int, float], float]]:
    """Select univariate polynomial degree and ridge penalty by CV MSE."""
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    scores = {}
    for degree in degrees:
        for lambda_ in lambdas:
            fold_mse = []
            for train_indices, validation_indices in _fold_indices(len(y), n_splits, random_state):
                x_train = X[train_indices, feature_index : feature_index + 1]
                x_validation = X[validation_indices, feature_index : feature_index + 1]
                mean, scale = _fit_standardizer(x_train)
                X_train = polynomial_features(_standardize(x_train, mean, scale), degree)
                X_validation = polynomial_features(_standardize(x_validation, mean, scale), degree)
                weights = fit_ridge(X_train, y[train_indices], lambda_)
                predictions = predict_linear(X_validation, weights)
                fold_mse.append(mean_squared_error(y[validation_indices], predictions))
            scores[(degree, float(lambda_))] = float(np.mean(fold_mse))

    best_degree, best_lambda = min(scores, key=scores.get)
    return best_degree, best_lambda, scores


def run_classification_experiment(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = RANDOM_SEED
) -> dict[str, object]:
    """Fit LDA and QDA on training data and evaluate each once on test data."""
    X_train, X_test, y_train, y_test = split_classification_data(X, y, test_size, random_state)
    mean, scale = _fit_standardizer(X_train)
    X_train = _standardize(X_train, mean, scale)
    X_test = _standardize(X_test, mean, scale)
    lda_model = fit_lda(X_train, y_train)
    qda_model = fit_qda(X_train, y_train)
    lda_predictions = predict_lda(X_test, *lda_model)
    qda_predictions = predict_qda(X_test, *qda_model)
    return {
        "n_train": len(y_train),
        "n_test": len(y_test),
        "lda": classification_metrics(y_test, lda_predictions),
        "qda": classification_metrics(y_test, qda_predictions),
    }


def run_wine_experiment(random_state: int = RANDOM_SEED) -> dict[str, object]:
    """Run the LDA/QDA evaluation workflow on the Wine dataset."""
    X, y = load_wine_data()
    return run_classification_experiment(X, y, random_state=random_state)


def run_regression_experiment(
    X: np.ndarray,
    y: np.ndarray,
    polynomial_feature_index: int = 2,
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS,
    degrees: tuple[int, ...] = DEFAULT_DEGREES,
    test_size: float = 0.2,
    n_splits: int = 5,
    random_state: int = RANDOM_SEED,
) -> dict[str, object]:
    """Tune on training folds, refit on training data, and evaluate once on test data."""
    X_train, X_test, y_train, y_test = split_regression_data(X, y, test_size, random_state)
    ridge_lambda, ridge_cv_mse = select_ridge_lambda(
        X_train, y_train, lambdas, n_splits, random_state
    )
    degree, polynomial_lambda, polynomial_cv_mse = select_polynomial_parameters(
        X_train,
        y_train,
        polynomial_feature_index,
        degrees,
        lambdas,
        n_splits,
        random_state,
    )

    mean, scale = _fit_standardizer(X_train)
    X_train_linear = _add_intercept(_standardize(X_train, mean, scale))
    X_test_linear = _add_intercept(_standardize(X_test, mean, scale))
    ols_predictions = predict_linear(X_test_linear, fit_ols(X_train_linear, y_train))
    ridge_weights = fit_ridge(X_train_linear, y_train, ridge_lambda)
    ridge_predictions = predict_linear(X_test_linear, ridge_weights)

    x_train = X_train[:, polynomial_feature_index : polynomial_feature_index + 1]
    x_test = X_test[:, polynomial_feature_index : polynomial_feature_index + 1]
    polynomial_mean, polynomial_scale = _fit_standardizer(x_train)
    X_train_polynomial = polynomial_features(
        _standardize(x_train, polynomial_mean, polynomial_scale), degree
    )
    X_test_polynomial = polynomial_features(
        _standardize(x_test, polynomial_mean, polynomial_scale), degree
    )
    polynomial_weights = fit_ridge(X_train_polynomial, y_train, polynomial_lambda)
    polynomial_predictions = predict_linear(X_test_polynomial, polynomial_weights)

    return {
        "n_train": len(y_train),
        "n_test": len(y_test),
        "ols": regression_metrics(y_test, ols_predictions),
        "ridge": {
            "lambda": ridge_lambda,
            "cv_mse": ridge_cv_mse,
            "test": regression_metrics(y_test, ridge_predictions),
        },
        "polynomial": {
            "feature_index": polynomial_feature_index,
            "degree": degree,
            "lambda": polynomial_lambda,
            "cv_mse": polynomial_cv_mse,
            "test": regression_metrics(y_test, polynomial_predictions),
        },
    }


def run_diabetes_experiment(random_state: int = RANDOM_SEED) -> dict[str, object]:
    """Run the OLS, ridge, and polynomial workflow on Diabetes data."""
    X, y = load_diabetes_data()
    return run_regression_experiment(X, y, random_state=random_state)
