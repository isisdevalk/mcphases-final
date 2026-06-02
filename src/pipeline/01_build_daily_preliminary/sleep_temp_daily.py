"""Daily sleep × wrist-temperature feature builder."""

import pandas as pd  # type: ignore
from pathlib import Path

RAW = Path("data/raw")
TEMP_FILE = RAW / "computed_temperature.csv"

print("\nBuilding DAILY temperature table\n")

# 1. Load data
temp = pd.read_csv(TEMP_FILE)

print("Raw shape:", temp.shape)

# 2. Align night to wake day
temp["day_in_study"] = temp["sleep_end_day_in_study"]

# 3. Keep core variable
temp_core = temp[[
    "id",
    "study_interval",
    "day_in_study",
    "nightly_temperature",
    "temperature_samples"
]]

# 4. Aggregate to daily
keys = ["id", "study_interval", "day_in_study"]

temp_daily = (
    temp_core
    .groupby(keys)
    .mean(numeric_only=True)
    .reset_index()
)

print("\nDaily temperature table created")
print("Shape:", temp_daily.shape)

print("\nMissingness:")
print(temp_daily.isna().mean().sort_values(ascending=False).head(10))

print("\nPreview:")
pd.set_option("display.max_columns", None)
print(temp_daily.head())

# Keep in memory if running interactively:
# python -i src/sleep_temp_daily.py