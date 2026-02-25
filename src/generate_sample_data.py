from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample_accidents(n_rows: int = 4000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    base_time = pd.Timestamp("2023-01-01")
    timestamps = base_time + pd.to_timedelta(rng.integers(0, 365 * 24, size=n_rows), unit="h")

    road_type = rng.choice(["urban", "highway", "rural"], size=n_rows, p=[0.55, 0.25, 0.20])
    weather = rng.choice(["clear", "rain", "fog"], size=n_rows, p=[0.72, 0.22, 0.06])
    lighting = rng.choice(["daylight", "night_lit", "night_unlit"], size=n_rows, p=[0.55, 0.30, 0.15])
    intersection_type = rng.choice(["none", "t_junction", "cross"], size=n_rows, p=[0.5, 0.3, 0.2])

    speed_limit = np.where(
        road_type == "highway", rng.choice([80, 100, 120], size=n_rows),
        np.where(road_type == "urban", rng.choice([30, 40, 50, 60], size=n_rows), rng.choice([50, 60, 80], size=n_rows))
    )

    traffic_volume = (
        np.where(road_type == "urban", rng.normal(15000, 3500, n_rows),
                 np.where(road_type == "highway", rng.normal(24000, 4500, n_rows), rng.normal(8000, 2400, n_rows)))
    ).clip(1500, None).astype(int)

    # Severity risk score (synthetic relationship)
    is_night_unlit = (lighting == "night_unlit").astype(int)
    is_rain = (weather == "rain").astype(int)
    is_high_speed = (speed_limit >= 80).astype(int)

    risk = -1.5 + 0.9 * is_night_unlit + 0.6 * is_rain + 0.75 * is_high_speed
    prob_severe = 1 / (1 + np.exp(-risk))
    severe_flag = rng.binomial(1, prob_severe)

    injury_severity = np.where(
        severe_flag == 1,
        rng.choice(["serious", "fatal"], size=n_rows, p=[0.82, 0.18]),
        rng.choice(["minor", "moderate"], size=n_rows, p=[0.65, 0.35]),
    )

    df = pd.DataFrame(
        {
            "accident_id": [f"A{i:06d}" for i in range(1, n_rows + 1)],
            "date_time": timestamps,
            "segment_id": rng.integers(1, 120, size=n_rows),
            "latitude": 28.45 + rng.normal(0, 0.07, size=n_rows),
            "longitude": 77.10 + rng.normal(0, 0.07, size=n_rows),
            "road_type": road_type,
            "intersection_type": intersection_type,
            "lanes": rng.choice([1, 2, 3, 4], size=n_rows, p=[0.15, 0.55, 0.2, 0.1]),
            "speed_limit": speed_limit,
            "traffic_volume": traffic_volume,
            "weather": weather,
            "visibility": rng.choice(["good", "moderate", "poor"], size=n_rows, p=[0.75, 0.18, 0.07]),
            "lighting": lighting,
            "vehicle_type": rng.choice(["car", "bike", "truck", "bus"], size=n_rows, p=[0.55, 0.24, 0.14, 0.07]),
            "driver_age_group": rng.choice(["18-25", "26-40", "41-60", "60+"], size=n_rows, p=[0.22, 0.40, 0.28, 0.10]),
            "injury_severity": injury_severity,
            "injuries": np.where(severe_flag == 1, rng.poisson(2, n_rows), rng.poisson(1, n_rows)),
            "fatalities": np.where(injury_severity == "fatal", rng.integers(1, 4, n_rows), 0),
        }
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic traffic accident dataset.")
    parser.add_argument("--rows", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/raw/accidents.csv"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = generate_sample_accidents(n_rows=args.rows, seed=args.seed)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
