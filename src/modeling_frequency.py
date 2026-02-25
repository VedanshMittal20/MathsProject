from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


FEATURES = ["avg_speed_limit", "vehicles_avg", "casualties_avg", "fatalities_avg", "rainy_frac", "night_frac", "alcohol_frac"]
TARGET = "accidents"


def fit_frequency_models(data_path: Path) -> dict:
    df = pd.read_csv(data_path)
    X = sm.add_constant(df[FEATURES])
    y = df[TARGET]

    poisson = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    nb = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit()

    return {
        "rows": int(len(df)),
        "poisson_aic": float(poisson.aic),
        "negative_binomial_aic": float(nb.aic),
        "poisson_coefficients": poisson.params.to_dict(),
        "negative_binomial_coefficients": nb.params.to_dict(),
    }


def main() -> None:
    data_path = Path("data/processed/frequency_dataset.csv")
    output_path = Path("outputs/model_metrics/frequency_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}. Run data_prep first.")

    metrics = fit_frequency_models(data_path)
    output_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved frequency model metrics to {output_path}")


if __name__ == "__main__":
    main()
