from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "severity_binary"
NUMERIC_FEATURES = [
    "speed_limit",
    "vehicles_involved",
    "casualties",
    "fatalities",
    "driver_age",
    "is_night",
    "is_rainy",
    "is_peak_hour",
    "is_weekend",
    "alcohol_flag",
]
CATEGORICAL_FEATURES = [
    "state_name",
    "city_name",
    "month",
    "day_of_week",
    "vehicle_type",
    "weather",
    "road_type",
    "road_condition",
    "lighting",
    "traffic_control",
    "driver_gender",
    "driver_license_status",
    "location_details",
]


def fit_severity_model(data_path: Path) -> dict:
    df = pd.read_csv(data_path)
    df = df.dropna(subset=[TARGET]).copy()

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1200, class_weight="balanced")),
        ]
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    return {
        "rows": int(len(df)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "classification_report": classification_report(y_test, preds, output_dict=True),
    }


def main() -> None:
    data_path = Path("data/processed/accidents_clean.csv")
    output_path = Path("outputs/model_metrics/severity_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}. Run data_prep first.")

    metrics = fit_severity_model(data_path)
    output_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved severity model metrics to {output_path}")


if __name__ == "__main__":
    main()
