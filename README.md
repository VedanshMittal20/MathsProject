# Traffic Flow and Road Safety Analysis Project

This repository now includes an **actual runnable implementation** (not only documentation) for traffic accident analysis with:
- data generation/loading,
- data cleaning + feature engineering,
- accident frequency modelling,
- accident severity modelling,
- metrics export for reporting.

## Project Files
- `Traffic_Flow_Road_Safety_Project_Guide.md` — conceptual step-by-step project guide.
- `src/generate_sample_data.py` — creates a synthetic accident dataset if you do not have one yet.
- `src/data_prep.py` — cleans data and creates processed datasets.
- `src/modeling_frequency.py` — fits Poisson + Negative Binomial count models.
- `src/modeling_severity.py` — fits Logistic Regression severity model.
- `src/run_pipeline.py` — end-to-end runner.

## Expected Input Data
Place your real dataset at:

`data/raw/accidents.csv`

Minimum useful columns:
- `accident_id, date_time, segment_id`
- `road_type, intersection_type, lanes, speed_limit`
- `traffic_volume, weather, visibility, lighting`
- `vehicle_type, driver_age_group, injury_severity, injuries, fatalities`

If the file is missing, `run_pipeline.py` automatically generates sample data.

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_pipeline.py
```

## Outputs
After running, you will get:
- `data/processed/accidents_clean.csv`
- `data/processed/frequency_dataset.csv`
- `outputs/model_metrics/frequency_metrics.json`
- `outputs/model_metrics/severity_metrics.json`

These outputs can be directly used in your report’s Results and Discussion sections.
