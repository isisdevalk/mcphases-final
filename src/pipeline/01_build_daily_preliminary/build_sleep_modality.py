"""Build a per-night sleep-modality table (long → short main sleep, naps)."""

import pandas as pd  # type: ignore
from pathlib import Path

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

keys = ["id", "study_interval", "day_in_study"]

sleep_daily = pd.read_csv(PROCESSED / "sleep_daily_with_phases.csv")
temp_daily = pd.read_csv(PROCESSED / "sleep_temperature_daily.csv")

sleep_score = pd.read_csv(RAW / "sleep_score.csv").drop_duplicates()
sleep_score = sleep_score[keys + ["overall_score", "restlessness", "resting_heart_rate"]].copy()
sleep_score = sleep_score.rename(columns={
    "overall_score": "sleep_score_overall",
    "restlessness": "sleep_score_restlessness",
    "resting_heart_rate": "sleep_score_resting_hr",
})

sleep_modality = (
    sleep_daily
    .merge(sleep_score, on=keys, how="left")
    .merge(temp_daily, on=keys, how="left")
)

sleep_modality.to_csv(PROCESSED / "sleep_modality.csv", index=False)
print("Saved data/processed/sleep_modality.csv")
print("Shape:", sleep_modality.shape)

# python -i src/build_sleep_modality.py