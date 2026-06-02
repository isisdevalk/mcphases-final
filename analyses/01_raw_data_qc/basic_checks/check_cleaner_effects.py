"""Quantify the effect of basic cleaning steps (deduplication, range filtering)
on raw heart-rate and related signals."""

import pandas as pd   # type: ignore
from pathlib import Path

raw = Path("data/raw")

# HEART RATE CHECK

print("\n=== HEART RATE CLEANER CHECK ===")

df_hr = pd.read_csv(raw / "heart_rate.csv")
df_hr = df_hr.drop_duplicates(
    subset=["id", "study_interval", "day_in_study", "timestamp"]
)

df_hr["bpm"] = pd.to_numeric(df_hr["bpm"], errors="coerce")

before_valid = df_hr["bpm"].notna().sum()

invalid_mask = (df_hr["bpm"] < 30) | (df_hr["bpm"] > 230)
n_invalid = invalid_mask.sum()

print("Total rows:", len(df_hr))
print("Invalid HR values (<30 or >230):", n_invalid)

if n_invalid > 0:
    print("\nExamples:")
    print(df_hr.loc[invalid_mask, ["id", "day_in_study", "bpm"]].head(20))


# WRIST TEMP CHECK

print("\n=== WRIST TEMPERATURE CLEANER CHECK ===")

df_temp = pd.read_csv(raw / "wrist_temperature.csv")
df_temp = df_temp.drop_duplicates(
    subset=["id", "study_interval", "day_in_study", "timestamp"]
)

col = "temperature_diff_from_baseline"
df_temp[col] = pd.to_numeric(df_temp[col], errors="coerce")

invalid_mask = (df_temp[col] < -10) | (df_temp[col] > 10)
n_invalid = invalid_mask.sum()

print("Total rows:", len(df_temp))
print("Invalid temp values (outside [-10, 10]):", n_invalid)

if n_invalid > 0:
    print("\nExamples:")
    print(df_temp.loc[invalid_mask, ["id", "day_in_study", col]].head(20))
