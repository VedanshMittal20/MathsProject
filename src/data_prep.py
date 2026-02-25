from __future__ import annotations

from pathlib import Path

import pandas as pd


SEVERITY_MAP = {"minor": 0, "moderate": 0, "serious": 1, "fatal": 1}


def load_and_clean(raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path)

    df = df.drop_duplicates(subset=["accident_id"]).copy()
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    df = df.dropna(subset=["date_time"])  # essential field

    numeric_cols = ["speed_limit", "traffic_volume", "lanes", "injuries", "fatalities"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    cat_cols = ["road_type", "intersection_type", "weather", "visibility", "lighting", "vehicle_type", "driver_age_group", "injury_severity"]
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")

    df = df[df["speed_limit"] >= 0]
    df = df[df["traffic_volume"] > 0]

    # Feature engineering
    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.day_name()
    df["month"] = df["date_time"].dt.month
    df["is_weekend"] = df["date_time"].dt.weekday.isin([5, 6]).astype(int)
    df["is_night"] = df["hour"].isin([20, 21, 22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    df["is_peak_hour"] = df["hour"].isin([8, 9, 10, 17, 18, 19]).astype(int)
    df["is_rainy"] = df["weather"].str.lower().eq("rain").astype(int)
    df["severity_binary"] = df["injury_severity"].map(SEVERITY_MAP).fillna(0).astype(int)
    df["date"] = df["date_time"].dt.date

    return df


def build_frequency_dataset(clean_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        clean_df.groupby(["segment_id", "date"], as_index=False)
        .agg(
            accidents=("accident_id", "count"),
            traffic_volume=("traffic_volume", "mean"),
            speed_limit=("speed_limit", "mean"),
            lanes=("lanes", "mean"),
            is_rainy=("is_rainy", "mean"),
            is_night=("is_night", "mean"),
            is_peak_hour=("is_peak_hour", "mean"),
        )
    )
    grouped["accidents_per_10k_vehicles"] = grouped["accidents"] / grouped["traffic_volume"] * 10_000
    return grouped


def save_processed_data(clean_df: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    accident_path = out_dir / "accidents_clean.csv"
    freq_path = out_dir / "frequency_dataset.csv"

    freq_df = build_frequency_dataset(clean_df)
    clean_df.to_csv(accident_path, index=False)
    freq_df.to_csv(freq_path, index=False)
    return accident_path, freq_path


def main() -> None:
    raw_path = Path("data/raw/accidents.csv")
    out_dir = Path("data/processed")

    if not raw_path.exists():
        raise FileNotFoundError(f"Input file not found: {raw_path}. Generate sample data first.")

    clean_df = load_and_clean(raw_path)
    accident_path, freq_path = save_processed_data(clean_df, out_dir)
    print(f"Saved cleaned accident dataset to {accident_path}")
    print(f"Saved frequency dataset to {freq_path}")


if __name__ == "__main__":
    main()
