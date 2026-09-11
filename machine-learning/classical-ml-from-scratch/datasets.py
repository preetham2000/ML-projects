"""Public dataset loaders and reproducible train/test split helpers."""

import numpy as np


RANDOM_SEED = 42
WINE_SOURCE = "https://archive.ics.uci.edu/dataset/109/wine"
DIABETES_SOURCE = "https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset"


def _load_wine():
    from sklearn.datasets import load_wine

    return load_wine()


def _load_diabetes():
    from sklearn.datasets import load_diabetes

    return load_diabetes()


def load_wine_data() -> tuple[np.ndarray, np.ndarray]:
    """Load UCI Wine chemical measurements and cultivar labels via scikit-learn."""
    dataset = _load_wine()
    return np.asarray(dataset.data, dtype=float), np.asarray(dataset.target).ravel()


def load_diabetes_data() -> tuple[np.ndarray, np.ndarray]:
    """Load scikit-learn's built-in Diabetes regression dataset."""
    dataset = _load_diabetes()
    X = np.asarray(dataset.data, dtype=float)
    y = np.asarray(dataset.target, dtype=float).ravel()
    return X, y


def split_classification_data(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = RANDOM_SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a reproducible stratified train/test split."""
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    generator = np.random.default_rng(random_state)
    train_indices = []
    test_indices = []
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        generator.shuffle(indices)
        n_test = min(len(indices) - 1, max(1, round(len(indices) * test_size)))
        test_indices.extend(indices[:n_test])
        train_indices.extend(indices[n_test:])

    train_indices = np.asarray(train_indices)
    test_indices = np.asarray(test_indices)
    generator.shuffle(train_indices)
    generator.shuffle(test_indices)
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def split_regression_data(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = RANDOM_SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a reproducible shuffled train/test split for regression."""
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    indices = np.random.default_rng(random_state).permutation(len(y))
    n_test = round(len(y) * test_size)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
