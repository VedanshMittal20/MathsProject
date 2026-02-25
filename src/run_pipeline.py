from __future__ import annotations

from pathlib import Path

from data_prep import load_and_clean, save_processed_data
from generate_sample_data import generate_sample_accidents
from modeling_frequency import fit_frequency_models
from modeling_severity import fit_severity_model


def main() -> None:
    raw_path = Path("data/raw/accidents.csv")
    processed_dir = Path("data/processed")
    metrics_dir = Path("outputs/model_metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        sample = generate_sample_accidents(n_rows=4000, seed=42)
        sample.to_csv(raw_path, index=False)
        print(f"No raw dataset found. Generated sample data at {raw_path}")

    clean_df = load_and_clean(raw_path)
    accident_path, freq_path = save_processed_data(clean_df, processed_dir)
    print(f"Prepared datasets: {accident_path}, {freq_path}")

    freq_metrics = fit_frequency_models(freq_path)
    sev_metrics = fit_severity_model(accident_path)

    (metrics_dir / "frequency_metrics.json").write_text(__import__("json").dumps(freq_metrics, indent=2))
    (metrics_dir / "severity_metrics.json").write_text(__import__("json").dumps(sev_metrics, indent=2))

    print("Pipeline complete.")
    print("Frequency metrics -> outputs/model_metrics/frequency_metrics.json")
    print("Severity metrics -> outputs/model_metrics/severity_metrics.json")


if __name__ == "__main__":
    main()
