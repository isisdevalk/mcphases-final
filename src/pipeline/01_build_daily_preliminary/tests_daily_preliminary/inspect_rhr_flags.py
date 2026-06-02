"""Inspect resting-heart-rate quality-control flags in the preliminary daily master."""

import pandas as pd  # type: ignore
from pathlib import Path

FLAGFILE = Path("data/processed/qc/rhr_outside_25_120.csv")

def main() -> None:
    if not FLAGFILE.exists():
        raise FileNotFoundError(f"Missing: {FLAGFILE}")

    df = pd.read_csv(FLAGFILE)

    if "rhr_value" not in df.columns:
        raise ValueError("Column 'rhr_value' not found in flagged file.")

    print("\nInspecting flagged RHR values\n")
    print("Rows flagged:", len(df))

    # Overall distribution
    print("\n--- rhr_value.describe() ---")
    print(df["rhr_value"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))

    # How many low vs high?
    low = df[df["rhr_value"] < 25]
    high = df[df["rhr_value"] > 120]

    print("\n--- Direction of flags ---")
    print("Low (<25):", len(low))
    print("High (>120):", len(high))

    # Show top extremes
    print("\n--- Lowest 10 rhr_value ---")
    print(df.sort_values("rhr_value")[["id", "study_interval", "day_in_study", "rhr_value"]].head(10))

    print("\n--- Highest 10 rhr_value ---")
    print(df.sort_values("rhr_value", ascending=False)[["id", "study_interval", "day_in_study", "rhr_value"]].head(10))

    # Optional: flag very extreme values (suggested next QC threshold)
    extreme_hi = df[df["rhr_value"] > 150]
    extreme_lo = df[df["rhr_value"] < 30]

    print("\n--- Very extreme (suggested investigate) ---")
    print("rhr_value > 150:", len(extreme_hi))
    print("rhr_value < 30:", len(extreme_lo))

if __name__ == "__main__":
    main()
