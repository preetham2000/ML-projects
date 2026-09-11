# Classical Machine Learning from Scratch

A compact NumPy implementation of several classical machine-learning algorithms, with reproducible experiments, public datasets, unit tests, and comparisons to scikit-learn reference implementations. The custom implementations in `models.py` are the models used by the experiments.

## Implemented from scratch

- Linear discriminant analysis (LDA)
- Quadratic discriminant analysis (QDA)
- Ordinary least squares (OLS)
- Ridge regression, including its analytic objective gradient
- Univariate polynomial feature generation and polynomial ridge regression

The implementations use NumPy linear algebra: least-squares solving for OLS, linear solves for ridge and discriminant scores, and log-space QDA scoring. They support arbitrary class labels, empirical or supplied class priors, and flat or column-vector regression targets where applicable.

## Datasets

| Task | Dataset | Source | Use in this project |
| --- | --- | --- | --- |
| Classification | Wine | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/109/wine), loaded locally through `sklearn.datasets.load_wine` | 178 wines, 13 chemical features, 3 cultivar classes |
| Regression | Diabetes | [scikit-learn Diabetes dataset documentation](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset), loaded with `sklearn.datasets.load_diabetes` | 442 observations, 10 baseline variables, disease-progression target after one year |

Both datasets are supplied by scikit-learn; running the project does not require downloading course data or accessing a network.

## Experiment methodology

All experiments use `random_state=42`.

- **Wine classification:** a stratified 80/20 train/test split (142/36 rows). Feature scaling is fit on training rows only; LDA and QDA are then fit on the transformed training data and evaluated once on the held-out test set.
- **Diabetes regression:** a shuffled 80/20 train/test split (354/88 rows). Feature scaling is fitted separately within each training fold. Five-fold cross-validation on the training partition selects ridge `lambda` from `0, 0.01, 0.1, 1, 10, 100`; the selected model is then refit on all training rows and evaluated once on test data.
- **Polynomial regression:** the same training-only cross-validation jointly selects degree `1`–`4` and `lambda` for a univariate BMI feature experiment. It is intentionally reported separately from the full-feature linear models.

## Results

Results below come from `python run_experiments.py` with the default seed and are also written to `outputs/results.json`.

| Task | Model | Selection | Held-out result |
| --- | --- | --- | --- |
| Wine | LDA | — | Accuracy **0.944**, balanced accuracy **0.952** |
| Wine | QDA | — | Accuracy **0.972**, balanced accuracy **0.967** |
| Diabetes | OLS | — | MSE **3502.242**, RMSE **59.180**, R² **0.385** |
| Diabetes | Ridge | 5-fold CV selected `lambda=10` (CV MSE 2841.413) | MSE **3501.820**, RMSE **59.176**, R² **0.385** |
| Diabetes | Polynomial ridge (BMI only) | 5-fold CV selected degree `1`, `lambda=0` | MSE **4168.977**, RMSE **64.568**, R² **0.268** |

The BMI-only polynomial model is a deliberately restricted univariate experiment and performs worse than the full-feature linear models on this split.

## Reference comparisons and tests

The custom models remain the primary implementations. Tests compare their behavior with established references while accounting for intentional covariance-normalization differences in LDA and QDA:

- LDA and QDA predictions against `sklearn.discriminant_analysis`
- OLS and ridge predictions against `sklearn.linear_model`
- Polynomial feature matrices against `sklearn.preprocessing.PolynomialFeatures`
- The analytic ridge gradient against a centered finite-difference gradient
- The closed-form ridge solution against `scipy.optimize.minimize`

LDA/QDA predictions agree with scikit-learn on the balanced, equal-prior fixture. An imbalanced boundary-point test documents that valid unbiased covariance estimates can yield different predictions from scikit-learn's normalization.

The suite also covers nonconsecutive class labels, rank-deficient OLS and zero-penalty ridge inputs, singular QDA covariance behavior, reproducible splits, training-only preprocessing, built-in dataset shapes, and both `(n,)` and `(n, 1)` target forms.

## Generated visualizations

Running the experiments regenerates these files in `outputs/`:

- `classification_wine_summary.png`: held-out LDA/QDA scores and a two-component, training-fitted PCA projection of the Wine data. The projection is explicitly not a decision boundary for the 13-feature models.
- `regression_diabetes_summary.png`: ridge cross-validation MSE across candidate penalties and a held-out Diabetes regression summary.
- `results.json`: machine-readable experiment metrics and selected hyperparameters.

## Project structure

```text
models.py             # NumPy implementations of the ML algorithms
datasets.py           # Public dataset loaders and reproducible split helpers
experiments.py        # Train/CV/test evaluation workflows
reporting.py          # Plot and JSON-output helpers
run_experiments.py    # Regenerates outputs/
tests/                # Model, dataset, workflow, reporting, and reference tests
original/             # Preserved coursework reference materials (unchanged)
```

## Setup

Tested with Python 3.13. The code requires Python 3.10 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the experiments

```bash
python run_experiments.py
```

This recreates the figures and `results.json` under `outputs/` using the fixed default seed.

## Run the tests

```bash
python -m unittest discover -s tests -v
```

## Technical takeaways

- Correct evaluation depends on separating the final test set from cross-validation and preprocessing.
- Numerically sound linear-algebra routines (`lstsq`, `solve`, and `slogdet`) avoid explicit matrix inversion and unstable probability calculations.
- Matching intercept, prior, covariance, and regularization conventions is essential when comparing independent implementations.
- A simpler, constrained model can be useful for inspection but should be evaluated against stronger baselines rather than assumed to improve performance.

## Project Origin

The core algorithms were originally implemented during graduate-level coursework. They were subsequently restructured into this independent project with public datasets, improved evaluation methodology, testing, and reference comparisons.
