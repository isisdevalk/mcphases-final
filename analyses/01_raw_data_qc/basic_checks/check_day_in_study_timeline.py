"""Inspect timeline properties (range, monotonicity, gaps) of `day_in_study`
across all raw CSVs that carry it."""

# analyses/check_day_in_study_timeline.py

import pandas as pd # type: ignore
from pathlib import Path

raw_dir = Path("data/raw")

print("\nChecking timeline properties for files with 'day_in_study':\n")

for csv_file in sorted(raw_dir.glob("*.csv")):
    df = pd.read_csv(csv_file, nrows=500)  # sample rows for speed

    if "day_in_study" not in df.columns or "id" not in df.columns:
        continue

    print(f"\n--- {csv_file.name} ---")

    # Basic properties
    print("Rows (sample):", len(df))
    print("day_in_study type:", df["day_in_study"].dtype)

    # Range per participant (first few)
    ranges = (
        df.groupby("id")["day_in_study"]
        .agg(["min", "max", "nunique"])
        .head()
    )
    print("Day range per participant (sample):")
    print(ranges)

    # Multiple rows per day?
    duplicates = (
        df.groupby(["id", "day_in_study"]).size().max()
    )
    print("Max rows per (id, day):", duplicates)


""""Daily (1 row per person per day -> use directly)
- hormones_and_selfreport.csv 
- active_minutes.csv
- resting_heart_rate.csv, 
- time_in_heart_rate_zones.csv
- demographic_vo2_max.csv (quasi-static but daily indexed)
- sleep_score.csv

High-frequency (many rows per day -> aggregate mean/median/max etc.)
- heart_rate.csv
- glucose.csv
- estimated_oxygen_variation.csv
- wrist_temperature.csv

Event to be added up to daily (few-many rows per day -> aggregate sum)
- steps.csv 
- calories.csv 
- distance.csv 
- exercise.csv 

Event / semi-daily (few-many rows per day -> aggregate carefully)
- altitude.csv
- heart_rate_variability_details.csv
- respiratory_rate_summary.csv
- stress_score.csv

Static (no daily timeline -> merge once)
- subject-info.csv
- height_and_weight.csv """