"""Investigate suspicious zero-valued rows in the raw HRV signals (RMSSD, HF, LF)."""

import pandas as pd     # type: ignore
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("results/hrv_checks")
OUT.mkdir(parents=True, exist_ok=True)

FILE = RAW / "heart_rate_variability_details.csv"

print("\nLoading HRV data...\n")

df = pd.read_csv(FILE)

# Columns to inspect
hrv_cols = ["rmssd", "high_frequency", "low_frequency"]

# Ensure numeric
for col in hrv_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("Basic summary statistics:\n")
print(df[hrv_cols].describe())

print("\nChecking zeros and near-zero values...\n")

report = []

for col in hrv_cols:
    total = df[col].notna().sum()
    zero_count = (df[col] == 0).sum()
    near_zero_count = (df[col] < 1).sum()  # threshold for suspiciously low values
    
    percent_zero = (zero_count / total * 100) if total > 0 else 0
    percent_near_zero = (near_zero_count / total * 100) if total > 0 else 0

    print(f"{col}:")
    print(f"  Total non-missing: {total}")
    print(f"  Exact zeros: {zero_count} ({percent_zero:.2f}%)")
    print(f"  Values < 1: {near_zero_count} ({percent_near_zero:.2f}%)")
    print(f"  Minimum value: {df[col].min()}")
    print("-" * 40)

    report.append({
        "variable": col,
        "total_non_missing": total,
        "zero_count": zero_count,
        "percent_zero": percent_zero,
        "near_zero_count": near_zero_count,
        "percent_near_zero": percent_near_zero,
        "min_value": df[col].min()
    })

# Save report
report_df = pd.DataFrame(report)
report_df.to_csv(OUT / "hrv_zero_check_report.csv", index=False)

print("\nReport saved to:", OUT / "hrv_zero_check_report.csv")

# Optional: save sample problematic rows
problem_rows = df[
    (df["rmssd"] == 0) |
    (df["high_frequency"] == 0) |
    (df["low_frequency"] == 0)
]

if len(problem_rows) > 0:
    sample = problem_rows.head(50)
    sample.to_csv(OUT / "hrv_zero_examples.csv", index=False)
    print("Sample zero rows saved to:", OUT / "hrv_zero_examples.csv")
else:
    print("No exact zero rows found.")

print("\nDone.")
