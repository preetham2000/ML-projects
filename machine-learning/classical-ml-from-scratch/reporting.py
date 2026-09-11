"""Plotting and JSON output helpers for portfolio experiments."""

import json
from pathlib import Path

import matplotlib
import numpy as np

from datasets import RANDOM_SEED, split_classification_data


matplotlib.use("Agg")
from matplotlib import pyplot as plt


def _as_json(value):
    """Convert NumPy values in experiment results to JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        if value and all(isinstance(key, tuple) and len(key) == 2 for key in value):
            return [
                {"degree": int(degree), "lambda": float(lambda_), "mse": _as_json(mse)}
                for (degree, lambda_), mse in value.items()
            ]
        return {str(key): _as_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_json(item) for item in value]
    return value


def write_results(results: dict[str, object], output_path: Path) -> None:
    """Write experiment results as formatted JSON."""
    output_path.write_text(json.dumps(_as_json(results), indent=2) + "\n")


def _wine_pca_projection(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project standardized Wine features onto two training-fitted principal components."""
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    train_standardized = (X_train - mean) / scale
    test_standardized = (X_test - mean) / scale
    _, singular_values, components = np.linalg.svd(train_standardized, full_matrices=False)
    explained_variance = singular_values**2 / (len(X_train) - 1)
    explained_ratio = explained_variance[:2] / explained_variance.sum()
    return train_standardized @ components[:2].T, test_standardized @ components[:2].T, explained_ratio


def save_classification_summary(
    results: dict[str, object], X: np.ndarray, y: np.ndarray, output_path: Path, random_state: int = RANDOM_SEED
) -> None:
    """Save Wine performance bars and a clearly labeled PCA projection."""
    X_train, X_test, _, y_test = split_classification_data(X, y, random_state=random_state)
    train_projection, test_projection, explained_ratio = _wine_pca_projection(X_train, X_test)
    model_names = ("LDA", "QDA")
    metrics = (results["lda"], results["qda"])
    positions = np.arange(len(model_names))

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(positions - 0.18, [metric["accuracy"] for metric in metrics], 0.36, label="Accuracy")
    axes[0].bar(
        positions + 0.18,
        [metric["balanced_accuracy"] for metric in metrics],
        0.36,
        label="Balanced accuracy",
    )
    axes[0].set_xticks(positions, model_names)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Wine held-out test performance")
    axes[0].legend()

    axes[1].scatter(train_projection[:, 0], train_projection[:, 1], color="lightgray", alpha=0.45, label="Train")
    for label in np.unique(y_test):
        mask = y_test == label
        axes[1].scatter(test_projection[mask, 0], test_projection[mask, 1], label=f"Test class {label}")
    axes[1].set_xlabel(f"PC1 ({explained_ratio[0]:.1%} variance)")
    axes[1].set_ylabel(f"PC2 ({explained_ratio[1]:.1%} variance)")
    axes[1].set_title("Wine test samples in a training-fitted PCA projection\nNot a model decision boundary")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_regression_summary(results: dict[str, object], output_path: Path) -> None:
    """Save ridge CV performance and a held-out regression metric summary."""
    ridge = results["ridge"]
    polynomial = results["polynomial"]
    cv_scores = ridge["cv_mse"]
    lambdas = np.array(sorted(float(value) for value in cv_scores))
    errors = np.array([cv_scores[float(value)] for value in lambdas])
    positions = np.arange(len(lambdas))
    selected_position = np.flatnonzero(lambdas == ridge["lambda"])[0]

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(positions, errors, marker="o")
    axes[0].axvline(selected_position, color="black", linestyle="--", label="Selected lambda")
    axes[0].set_xticks(positions, [f"{value:g}" for value in lambdas])
    axes[0].set_xlabel("Ridge lambda")
    axes[0].set_ylabel("Mean cross-validation MSE")
    axes[0].set_title("Diabetes ridge selection on training folds")
    axes[0].legend()

    axes[1].axis("off")
    summary = ["Held-out Diabetes test performance", "", "Model          MSE       RMSE      R²"]
    for name, metrics in (("OLS", results["ols"]), ("Ridge", ridge["test"]), ("Polynomial", polynomial["test"])):
        summary.append(f"{name:<12} {metrics['mse']:8.3f} {metrics['rmse']:8.3f} {metrics['r2']:7.3f}")
    summary.extend(
        [
            "",
            f"Ridge lambda: {ridge['lambda']}",
            f"Polynomial: BMI, degree {polynomial['degree']}, lambda {polynomial['lambda']}",
        ]
    )
    axes[1].text(0.02, 0.98, "\n".join(summary), va="top", family="monospace", fontsize=10)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
