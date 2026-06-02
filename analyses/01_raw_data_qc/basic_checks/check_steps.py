"""Sanity checks on raw steps.csv (negatives, extreme values, daily distribution)."""

import pandas as pd # type: ignore
from pathlib import Path

RAW = Path("data/raw")

print("\nLoading raw steps data...\n")

df = pd.read_csv(RAW / "steps.csv")

df["steps"] = pd.to_numeric(df["steps"], errors="coerce")

print("Basic summary:\n")
print(df["steps"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))

print("\nChecking invalid raw values...\n")

# 1 Negative values
neg = df[df["steps"] < 0]
print("Negative values:", len(neg))

# 2 Extremely large per-interval values
# For wearable minute-level data, >5000 steps in one interval is already extreme.
extreme = df[df["steps"] > 5000]
print("Rows > 5000 steps in a single record:", len(extreme))

# 3 Very extreme values
very_extreme = df[df["steps"] > 20000]
print("Rows > 20000 steps in a single record:", len(very_extreme))

# Save examples if they exist
OUT = Path("results/activity_checks")
OUT.mkdir(parents=True, exist_ok=True)

if len(extreme) > 0:
 extreme.head(50).to_csv(OUT / "raw_steps_extreme_examples.csv", index=False)
 print("Saved extreme examples.")

if len(neg) > 0:
 neg.head(50).to_csv(OUT / "raw_steps_negative_examples.csv", index=False)
 print("Saved negative examples.")

print("\nDone.")


RAW = Path("data/raw")

df = pd.read_csv(RAW / "steps.csv")
df["steps"] = pd.to_numeric(df["steps"], errors="coerce")

# Extreme threshold
extreme = df[df["steps"] > 20000].copy()

print("\nNumber of extreme raw rows (>20k):", len(extreme))

if len(extreme) > 0:
 print("\nExtreme rows summary:")
 print(extreme[["id", "study_interval", "day_in_study"]].value_counts().head(20))

 print("\nExtreme rows by ID:")
 print(extreme["id"].value_counts())

 print("\nTop extreme values:")
 print(extreme.sort_values("steps", ascending=False).head(20))

 # Save full set
 OUT = Path("results/activity_checks")
 OUT.mkdir(parents=True, exist_ok=True)
 extreme.to_csv(OUT / "raw_steps_extreme_full.csv", index=False)
 print("\nSaved full extreme rows to results/activity_checks/raw_steps_extreme_full.csv")

extreme_5k = df[df["steps"] > 5000].copy()

print("Rows > 5000:", len(extreme_5k))

print("\nBy ID:")
print(extreme_5k["id"].value_counts())

print("\nBy (id, day):")
print(extreme_5k[["id", "study_interval", "day_in_study"]]
 .value_counts()
 .head(20))

print("\nChecking duplicate rows (full row duplicates)...")
dup_full = df.duplicated().sum()
print("Exact duplicate rows:", dup_full)

print("\nChecking duplicates per (id, interval, day, timestamp)...")
dup_time = df.duplicated(subset=["id", "study_interval", "day_in_study", "timestamp"]).sum()
print("Duplicate timestamp rows:", dup_time)
