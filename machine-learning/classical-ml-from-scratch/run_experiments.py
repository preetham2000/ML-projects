"""Generate reproducible portfolio outputs from the public datasets."""

from pathlib import Path

from datasets import RANDOM_SEED, load_diabetes_data, load_wine_data
from experiments import run_classification_experiment, run_regression_experiment
from reporting import save_classification_summary, save_regression_summary, write_results


def generate_outputs(output_directory: str | Path = "outputs", random_state: int = RANDOM_SEED) -> dict[str, object]:
    """Run both experiments and regenerate their figures and JSON results."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    wine_X, wine_y = load_wine_data()
    classification = run_classification_experiment(wine_X, wine_y, random_state=random_state)
    diabetes_X, diabetes_y = load_diabetes_data()
    regression = run_regression_experiment(diabetes_X, diabetes_y, random_state=random_state)
    results = {"random_seed": random_state, "classification": classification, "regression": regression}

    save_classification_summary(
        classification, wine_X, wine_y, output_path / "classification_wine_summary.png", random_state
    )
    save_regression_summary(regression, output_path / "regression_diabetes_summary.png")
    write_results(results, output_path / "results.json")
    return results


if __name__ == "__main__":
    results = generate_outputs()
    print(f"Wrote outputs for seed {results['random_seed']} to outputs/")
