"""Build daily sleep + temperature features from raw mcPHASES exports."""

import pandas as pd  # type: ignore
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

print("\nBuilding DAILY sleep temperature table\n")

# 1. Load data
temp = pd.read_csv(RAW / "computed_temperature.csv").drop_duplicates()

print("Raw shape:", temp.shape)

# 2. Align night to wake day
temp["day_in_study"] = temp["sleep_end_day_in_study"]

# 3. Keep core variables
temp_core = temp[[
    "id",
    "study_interval",
    "day_in_study",
    "nightly_temperature",
    "temperature_samples"
]].copy()

# 4. Aggregate to nightly (mean if multiple records)
keys = ["id", "study_interval", "day_in_study"]

temp_daily = (
    temp_core
    .groupby(keys)
    .mean(numeric_only=True)
    .reset_index()
)

# 5. Rename for clarity
temp_daily = temp_daily.rename(columns={
    "nightly_temperature": "sleep_temp_nightly",
    "temperature_samples": "sleep_temp_samples"
})

print("\nDaily temperature table created")
print("Shape:", temp_daily.shape)

print("\nMissingness:")
print(temp_daily.isna().mean().sort_values(ascending=False).head(10))

print("\nPreview:")
pd.set_option("display.max_columns", None)
print(temp_daily.head())

# 6. Save
out_path = OUT / "sleep_temperature_daily.csv"
temp_daily.to_csv(out_path, index=False)
print("\nSaved:", out_path)

# python -i src/build_sleep_temperature.py