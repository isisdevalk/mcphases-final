"""Sanity checks on raw distance.csv (range, duplicates, daily totals)."""

import pandas as pd # type: ignore
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("results/activity_checks")
OUT.mkdir(parents=True, exist_ok=True)

print("\nLoading raw distance data...\n")

df = pd.read_csv(RAW / "distance.csv")

# Ensure numeric
df["distance"] = pd.to_numeric(df["distance"], errors="coerce")

print("Basic summary:\n")
print(df["distance"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))

# 1 Duplicate checks

print("\nChecking duplicate rows...")

dup_full = df.duplicated().sum()
print("Exact duplicate rows:", dup_full)

dup_time = df.duplicated(
 subset=["id", "study_interval", "day_in_study", "timestamp"]
).sum()
print("Duplicate timestamp rows:", dup_time)

# 2 Negative values

neg = df[df["distance"] < 0]
print("\nNegative values:", len(neg))

# 3 Extremely large per-interval values

print("\nExtreme per-interval distance values:")

extreme_1 = df[df["distance"] > 10000] # threshold adjustable
print("Rows > 10,000:", len(extreme_1))

extreme_2 = df[df["distance"] > 50000]
print("Rows > 50,000:", len(extreme_2))

# Save examples if they exist
if len(extreme_2) > 0:
 extreme_2.head(50).to_csv(
 OUT / "raw_distance_extreme_examples.csv",
 index=False
 )
 print("Saved extreme examples.")

# 4 Daily aggregation BEFORE cleaning

daily = (
 df.groupby(["id", "study_interval", "day_in_study"])["distance"]
 .sum()
 .reset_index(name="distance_daily")
)

print("\nDaily distance summary (before cleaning):")
print(daily["distance_daily"].describe(percentiles=[0.95, 0.99]))

print("\nTop 10 daily distances:")
print(daily.sort_values("distance_daily", ascending=False).head(10))

# 5 Save daily for inspection

daily.to_csv(OUT / "distance_daily_raw_check.csv", index=False)

print("\nDone. Results saved to:", OUT)
