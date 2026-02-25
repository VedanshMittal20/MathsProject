from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_prep import load_and_clean, save_processed_data


def export_numeric(clean_df: pd.DataFrame, out_path: Path) -> Path:
    """
    One-hot encode categoricals so Scilab can read a pure numeric matrix.
    """
    target = clean_df["severity_binary"]
    features = clean_df.drop(columns=["severity_binary"])

    cat_cols = features.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in features.columns if c not in cat_cols]

    encoded = pd.get_dummies(features, columns=cat_cols, drop_first=True)
    encoded.insert(0, "severity_binary", target.values)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoded.to_csv(out_path, index=False)
    return out_path


def export_core_numeric(clean_df: pd.DataFrame, out_path: Path) -> Path:
    """
    Small, purely numeric subset for quick Scilab experimentation.
    """
    cols = [
        "severity_binary",
        "speed_limit",
        "vehicles_involved",
        "casualties",
        "fatalities",
        "driver_age",
        "hour",
        "month_num",
        "is_night",
        "is_peak_hour",
        "is_rainy",
        "is_weekend",
        "alcohol_flag",
    ]
    subset = clean_df[cols].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    raw_path = Path("accident_prediction_india.csv")
    processed_dir = Path("data/processed")
    numeric_out = processed_dir / "accidents_clean_numeric.csv"
    core_out = processed_dir / "accidents_core_numeric.csv"
    freq_numeric_out = processed_dir / "frequency_dataset_numeric.csv"

    if not raw_path.exists():
        raise FileNotFoundError(f"Input file not found: {raw_path}")

    clean_df = load_and_clean(raw_path)
    accident_path, freq_path = save_processed_data(clean_df, processed_dir)  # keeps frequency + clean CSV

    numeric_path = export_numeric(clean_df, numeric_out)
    core_path = export_core_numeric(clean_df, core_out)

    freq_df = pd.read_csv(freq_path)
    freq_df["state_code"] = freq_df["state_name"].astype("category").cat.codes
    freq_numeric_cols = [
        "state_code",
        "month_num",
        "accidents",
        "avg_speed_limit",
        "vehicles_avg",
        "casualties_avg",
        "fatalities_avg",
        "rainy_frac",
        "night_frac",
        "alcohol_frac",
    ]
    freq_df[freq_numeric_cols].to_csv(freq_numeric_out, index=False)

    print(f"Exported Scilab-friendly numeric dataset to {numeric_path}")
    print(f"Exported compact numeric dataset to {core_path}")
    print(f"Exported numeric frequency dataset to {freq_numeric_out}")


if __name__ == "__main__":
    main()
