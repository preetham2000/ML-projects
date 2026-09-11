"""From-scratch classical machine-learning routines."""

import numpy as np


def _resolve_priors(counts: np.ndarray, priors: np.ndarray | None) -> np.ndarray:
    """Return empirical priors or supplied priors ordered by class labels."""
    if priors is None:
        return counts / counts.sum()
    return np.asarray(priors, dtype=float)


def fit_lda(
    X: np.ndarray, y: np.ndarray, priors: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Learn class labels, means, pooled covariance, and class priors for LDA."""
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    n_samples, n_features = X.shape
    classes, counts = np.unique(y, return_counts=True)
    means = []
    covariance = np.zeros((n_features, n_features))
    for class_label in classes:
        X_class = X[y == class_label]
        mean = np.mean(X_class, axis=0)
        covariance += (X_class - mean).T @ (X_class - mean)
        means.append(mean)

    covariance /= n_samples - len(classes)
    return classes, np.asarray(means).T, covariance, _resolve_priors(counts, priors)


def fit_qda(
    X: np.ndarray, y: np.ndarray, priors: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    """Learn class labels, means, covariances, and class priors for QDA."""
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    classes, counts = np.unique(y, return_counts=True)
    means = []
    covariances = []
    for class_label in classes:
        X_class = X[y == class_label]
        mean = np.mean(X_class, axis=0)
        covariances.append((X_class - mean).T @ (X_class - mean) / (len(X_class) - 1))
        means.append(mean)

    return classes, np.asarray(means).T, covariances, _resolve_priors(counts, priors)


def predict_lda(
    X: np.ndarray,
    classes: np.ndarray,
    means: np.ndarray,
    covariance: np.ndarray,
    priors: np.ndarray,
) -> np.ndarray:
    """Return LDA predictions using the supplied fitted parameters."""
    X = np.asarray(X)
    scores = np.empty((len(X), len(classes)))
    for index, mean in enumerate(means.T):
        differences = X - mean
        solved = np.linalg.solve(covariance, differences.T).T
        scores[:, index] = -0.5 * np.sum(differences * solved, axis=1) + np.log(priors[index])
    return classes[np.argmax(scores, axis=1)]


def predict_qda(
    X: np.ndarray,
    classes: np.ndarray,
    means: np.ndarray,
    covariances: list[np.ndarray],
    priors: np.ndarray,
) -> np.ndarray:
    """Return QDA predictions using log Gaussian-discriminant scores."""
    X = np.asarray(X)
    scores = np.empty((len(X), len(classes)))
    for index, (mean, covariance) in enumerate(zip(means.T, covariances)):
        sign, log_determinant = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise np.linalg.LinAlgError("QDA covariance must have a positive determinant.")
        differences = X - mean
        solved = np.linalg.solve(covariance, differences.T).T
        scores[:, index] = (
            -0.5 * np.sum(differences * solved, axis=1)
            - 0.5 * log_determinant
            + np.log(priors[index])
        )
    return classes[np.argmax(scores, axis=1)]


def classification_accuracy(y_true: np.ndarray, predictions: np.ndarray) -> float:
    """Return the fraction of correctly predicted class labels."""
    return float(np.mean(np.asarray(y_true).ravel() == np.asarray(predictions).ravel()))


def _penalty_mask(n_features: int, penalize_intercept: bool) -> np.ndarray:
    """Return ridge penalty weights for a design matrix with a leading intercept."""
    mask = np.ones(n_features)
    if not penalize_intercept:
        mask[0] = 0
    return mask


def ridge_objective(
    weights: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    lambda_: float,
    penalize_intercept: bool = False,
) -> tuple[float, np.ndarray]:
    """Return ridge squared-error objective and analytic gradient."""
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    weights = np.asarray(weights).ravel()
    penalty = _penalty_mask(len(weights), penalize_intercept)
    residuals = y - X @ weights

    error = np.sum(residuals**2) + lambda_ * np.sum((penalty * weights) ** 2)
    gradient = -2 * X.T @ residuals + 2 * lambda_ * penalty * weights
    return float(error), gradient


def predict_linear(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return linear-model predictions."""
    return np.asarray(X) @ np.asarray(weights).ravel()


def mean_squared_error(y_true: np.ndarray, predictions: np.ndarray) -> float:
    """Return mean squared error for predictions."""
    residuals = np.asarray(y_true).ravel() - np.asarray(predictions).ravel()
    return float(np.mean(residuals**2))


def fit_ridge(
    X: np.ndarray, y: np.ndarray, lambda_: float, penalize_intercept: bool = False
) -> np.ndarray:
    """Learn ridge weights; by default, the leading intercept is unpenalized."""
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    if lambda_ == 0:
        return fit_ols(X, y)
    penalty = np.diag(_penalty_mask(X.shape[1], penalize_intercept))
    return np.linalg.solve(X.T @ X + lambda_ * penalty, X.T @ y)


def fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Learn ordinary least-squares weights using a least-squares solve."""
    return np.linalg.lstsq(np.asarray(X), np.asarray(y).ravel(), rcond=None)[0]


def polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    """Map a one-dimensional feature to powers from zero through ``degree``."""
    x = np.asarray(x).reshape(-1)
    return np.column_stack([x**power for power in range(degree + 1)])
