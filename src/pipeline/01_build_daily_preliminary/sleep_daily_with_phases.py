"""Attach menstrual-cycle phase labels to the daily sleep table."""

import pandas as pd  # type: ignore
import json
from pathlib import Path

RAW = Path("data/raw")
SLEEP_FILE = RAW / "sleep.csv"

print("\nBuilding DAILY sleep table WITH sleep phases\n")

# 1. Load data
sleep = pd.read_csv(SLEEP_FILE)

print("Raw shape:", sleep.shape)

# 2. Align night to wake day
sleep["day_in_study"] = sleep["sleep_end_day_in_study"]

# 3. Extract sleep stage minutes from 'levels'
def extract_minutes(levels_str, stage):
    try:
        levels = json.loads(levels_str.replace("'", '"'))
        return levels["summary"][stage]["minutes"]
    except Exception:
        return None

sleep["deep_minutes"] = sleep["levels"].apply(lambda x: extract_minutes(x, "deep"))
sleep["light_minutes"] = sleep["levels"].apply(lambda x: extract_minutes(x, "light"))
sleep["rem_minutes"] = sleep["levels"].apply(lambda x: extract_minutes(x, "rem"))
sleep["wake_minutes"] = sleep["levels"].apply(lambda x: extract_minutes(x, "wake"))

# 4. Convert duration to hours
sleep["sleep_hours"] = sleep["duration"] / (1000 * 60 * 60)

# 5. Create proportions (relative to total sleep)
sleep["deep_pct"] = sleep["deep_minutes"] / sleep["minutesasleep"]
sleep["rem_pct"] = sleep["rem_minutes"] / sleep["minutesasleep"]

# 6. Select core variables
sleep_core = sleep[[
    "id",
    "study_interval",
    "day_in_study",
    "sleep_hours",
    "minutesasleep",
    "minutesawake",
    "timeinbed",
    "efficiency",
    "deep_minutes",
    "light_minutes",
    "rem_minutes",
    "wake_minutes",
    "deep_pct",
    "rem_pct"
]]

# 7. Aggregate to daily level
keys = ["id", "study_interval", "day_in_study"]

sleep_daily = (
    sleep_core
    .groupby(keys)
    .mean(numeric_only=True)
    .reset_index()
)

print("\nDaily sleep table created")
print("Shape:", sleep_daily.shape)

print("\nMissingness (top 10):")
print(sleep_daily.isna().mean().sort_values(ascending=False).head(10))

pd.set_option("display.max_columns", None)
print("\nPreview:")
print(sleep_daily.head())

from pathlib import Path
Path("data/processed").mkdir(parents=True, exist_ok=True)
sleep_daily.to_csv("data/processed/sleep_daily_with_phases.csv", index=False)
print("Saved data/processed/sleep_daily_with_phases.csv")


# If running interactively:
# python -i src/sleep_daily_with_phases.py
