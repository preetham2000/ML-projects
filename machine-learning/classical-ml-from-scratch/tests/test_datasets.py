import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from datasets import (
    DIABETES_SOURCE,
    RANDOM_SEED,
    WINE_SOURCE,
    load_diabetes_data,
    load_wine_data,
    split_classification_data,
    split_regression_data,
)


class DatasetLoaderTests(unittest.TestCase):
    def test_wine_loader_returns_numeric_features_and_flat_labels(self):
        dataset = SimpleNamespace(data=[[1, 2], [3, 4]], target=[[0], [1]])

        with patch("datasets._load_wine", return_value=dataset) as load_wine:
            X, y = load_wine_data()

        load_wine.assert_called_once_with()
        np.testing.assert_array_equal(X, [[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(y, [0, 1])

    def test_diabetes_loader_returns_numeric_features_and_flat_targets(self):
        dataset = SimpleNamespace(data=[[1, 2], [3, 4]], target=[[5.0], [6.0]])

        with patch("datasets._load_diabetes", return_value=dataset) as load_diabetes:
            X, y = load_diabetes_data()

        load_diabetes.assert_called_once_with()
        np.testing.assert_array_equal(X, [[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(y, [5.0, 6.0])

    def test_documented_dataset_sources_are_public_urls(self):
        self.assertTrue(WINE_SOURCE.startswith("https://archive.ics.uci.edu/"))
        self.assertTrue(DIABETES_SOURCE.startswith("https://scikit-learn.org/"))

    def test_builtin_loaders_return_expected_dataset_shapes(self):
        wine_X, wine_y = load_wine_data()
        diabetes_X, diabetes_y = load_diabetes_data()

        self.assertEqual(wine_X.shape, (178, 13))
        self.assertEqual(len(np.unique(wine_y)), 3)
        self.assertEqual(diabetes_X.shape, (442, 10))
        self.assertEqual(diabetes_y.shape, (442,))


class DatasetSplitTests(unittest.TestCase):
    def test_classification_split_is_stratified_and_reproducible(self):
        X = np.arange(24).reshape(12, 2)
        y = np.repeat([10, 20, 30], 4)

        first_split = split_classification_data(X, y, test_size=0.25)
        second_split = split_classification_data(X, y, test_size=0.25)
        X_train, X_test, y_train, y_test = first_split

        for first, second in zip(first_split, second_split):
            np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(np.unique(y_train), [10, 20, 30])
        np.testing.assert_array_equal(np.unique(y_test), [10, 20, 30])
        np.testing.assert_array_equal(np.unique(y_test, return_counts=True)[1], [1, 1, 1])
        self.assertEqual(len(X_train) + len(X_test), len(X))

    def test_regression_split_is_reproducible_and_exhaustive(self):
        X = np.column_stack([np.arange(10), np.arange(10) ** 2])
        y = np.arange(10).reshape(-1, 1)

        first_split = split_regression_data(X, y, random_state=RANDOM_SEED)
        second_split = split_regression_data(X, y, random_state=RANDOM_SEED)
        X_train, X_test, y_train, y_test = first_split

        for first, second in zip(first_split, second_split):
            np.testing.assert_array_equal(first, second)
        self.assertEqual(len(X_train), 8)
        self.assertEqual(len(X_test), 2)
        np.testing.assert_array_equal(np.sort(np.concatenate([y_train, y_test])), np.arange(10))


if __name__ == "__main__":
    unittest.main()
