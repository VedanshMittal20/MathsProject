from __future__ import annotations

from pathlib import Path

import pandas as pd


SEVERITY_MAP = {"minor": 0, "moderate": 0, "serious": 1, "fatal": 1, "serious injury": 1}
INDIA_SEVERITY_MAP = {"minor": 0, "serious": 1, "fatal": 1}


def _safe_hour(value: str) -> float | None:
    """
    Parse hour from strings like '18:24' or '3:5'. Returns integer hour or None.
    """
    try:
        parts = str(value).split(":")
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
        if 0 <= hour < 24 and 0 <= minute < 60:
            return hour
    except Exception:
        return None
    return None


def load_and_clean(raw_path: Path) -> pd.DataFrame:
    """
    Cleaning routine tailored for the accident_prediction_india.csv schema.
    """
    df = pd.read_csv(raw_path)

    rename_map = {
        "State Name": "state_name",
        "City Name": "city_name",
        "Year": "year",
        "Month": "month",
        "Day of Week": "day_of_week",
        "Time of Day": "time_of_day",
        "Accident Severity": "accident_severity",
        "Number of Vehicles Involved": "vehicles_involved",
        "Vehicle Type Involved": "vehicle_type",
        "Number of Casualties": "casualties",
        "Number of Fatalities": "fatalities",
        "Weather Conditions": "weather",
        "Road Type": "road_type",
        "Road Condition": "road_condition",
        "Lighting Conditions": "lighting",
        "Traffic Control Presence": "traffic_control",
        "Speed Limit (km/h)": "speed_limit",
        "Driver Age": "driver_age",
        "Driver Gender": "driver_gender",
        "Driver License Status": "driver_license_status",
        "Alcohol Involvement": "alcohol_involvement",
        "Accident Location Details": "location_details",
    }
    df = df.rename(columns=rename_map)

    df["accident_id"] = df.index + 1  # synthetic unique id

    # Numeric columns
    numeric_cols = ["year", "speed_limit", "vehicles_involved", "casualties", "fatalities", "driver_age"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    # Categorical columns
    cat_cols = [
        "state_name",
        "city_name",
        "month",
        "day_of_week",
        "time_of_day",
        "accident_severity",
        "vehicle_type",
        "weather",
        "road_type",
        "road_condition",
        "lighting",
        "traffic_control",
        "driver_gender",
        "driver_license_status",
        "alcohol_involvement",
        "location_details",
    ]
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")

    # Month number for ordering
    df["month_num"] = pd.to_datetime(df["month"], format="%B", errors="coerce").dt.month
    df["month_num"] = df["month_num"].fillna(df["month_num"].mode().iloc[0])

    # Time features
    df["hour"] = df["time_of_day"].apply(_safe_hour)
    df["hour"] = df["hour"].fillna(df["hour"].mode().iloc[0])
    df["is_night"] = df["hour"].isin([20, 21, 22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    df["is_peak_hour"] = df["hour"].isin([8, 9, 10, 17, 18, 19]).astype(int)

    # Weather / alcohol flags
    df["is_rainy"] = df["weather"].str.lower().eq("rainy").astype(int)
    df["alcohol_flag"] = df["alcohol_involvement"].str.lower().eq("yes").astype(int)

    # Weekend flag from day name
    df["is_weekend"] = df["day_of_week"].str.lower().isin(["saturday", "sunday"]).astype(int)

    # Severity target (minor vs serious/fatal)
    df["severity_binary"] = (
        df["accident_severity"].str.lower().map(INDIA_SEVERITY_MAP).fillna(0).astype(int)
    )

    return df


def build_frequency_dataset(clean_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        clean_df.groupby(["state_name", "month_num"], as_index=False)
        .agg(
            accidents=("accident_id", "count"),
            avg_speed_limit=("speed_limit", "mean"),
            vehicles_avg=("vehicles_involved", "mean"),
            casualties_avg=("casualties", "mean"),
            fatalities_avg=("fatalities", "mean"),
            rainy_frac=("is_rainy", "mean"),
            night_frac=("is_night", "mean"),
            alcohol_frac=("alcohol_flag", "mean"),
        )
    )
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
