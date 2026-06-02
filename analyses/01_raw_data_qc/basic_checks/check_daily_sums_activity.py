"""Verify that within-day sums of activity signals (steps, distance, calories,
active zone minutes) reconcile with reported daily totals."""

# src/check_daily_sums_activity.py
import pandas as pd # type: ignore
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("results/activity_checks")
OUT.mkdir(parents=True, exist_ok=True)

KEYS = ["id", "study_interval", "day_in_study"]

SUM_FILES = {
 "active_zone_minutes.csv": "azm",
 "calories.csv": "cal",
 "distance.csv": "dist",
 "steps.csv": "steps",
}

def load_and_daily_sum(csv_name: str, prefix: str) -> pd.DataFrame:
 df = pd.read_csv(RAW / csv_name)

 # Drop non-analytic cols if present
 drop_cols = [c for c in ["is_weekend", "timestamp"] if c in df.columns]
 if drop_cols:
 df = df.drop(columns=drop_cols)

 # Ensure key cols exist
 missing = [k for k in KEYS if k not in df.columns]
 if missing:
 raise ValueError(f"{csv_name} missing key columns: {missing}")

 # Daily sum over numeric columns
 daily = df.groupby(KEYS).sum(numeric_only=True).reset_index()

 # Prefix non-key columns
 rename = {c: f"{prefix}_{c}" for c in daily.columns if c not in KEYS}
 daily = daily.rename(columns=rename)

 return daily


def check_ranges(daily_df: pd.DataFrame, prefix: str) -> None:
 cols = [c for c in daily_df.columns if c not in KEYS]
 print(f"\n--- {prefix}: daily summed columns ({len(cols)}) ---")
 for c in cols:
 s = daily_df[c]
 if s.notna().sum() == 0:
 print(f"{c}: all missing")
 continue
 desc = s.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
 print(f"\n{c}")
 print(desc)

 # quick sanity flags (do not remove automatically)
 if (s < 0).any():
 print(" Found negative values")
 if c.endswith("steps") and (s > 200000).any():
 print(" Extremely high steps days (>200k) present")
 if "distance" in c and (s > 200).any():
 print(" Extremely high distance days (>200) present (check units)")


def main():
 print("\nChecking daily SUM aggregation for activity/exposure files...\n")

 merged = None

 for fname, prefix in SUM_FILES.items():
 print(f"Processing {fname} → daily sum")
 daily = load_and_daily_sum(fname, prefix)

 # Save per-file daily summary
 out_csv = OUT / f"{prefix}_daily_sum.csv"
 daily.to_csv(out_csv, index=False)
 print("Saved:", out_csv)

 # Print distribution summaries
 check_ranges(daily, prefix)

 # Merge into one combined table for coverage checks
 merged = daily if merged is None else merged.merge(daily, on=KEYS, how="outer")

 # Coverage / missingness overview
 print("\n=== Coverage summary (merged) ===")
 print("Rows (unique id-interval-day):", len(merged))

 non_key_cols = [c for c in merged.columns if c not in KEYS]
 coverage = (
 merged[non_key_cols]
 .notna()
 .mean()
 .sort_values(ascending=False)
 .rename("fraction_non_missing")
 .reset_index()
 .rename(columns={"index": "column"})
 )

 out_cov = OUT / "coverage_fraction_non_missing.csv"
 coverage.to_csv(out_cov, index=False)
 print("Saved coverage:", out_cov)

 print("\nTop 15 columns by coverage:")
 print(coverage.head(15).to_string(index=False))

 print("\nBottom 15 columns by coverage:")
 print(coverage.tail(15).to_string(index=False))

 print("\nDone.\nAll outputs in:", OUT)


if __name__ == "__main__":
 main()
