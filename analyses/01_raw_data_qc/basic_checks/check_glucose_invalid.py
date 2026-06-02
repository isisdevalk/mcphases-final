#!/usr/bin/env python3
"""
Build daily glucose summaries (mean + median), perform minimal invalid-value filtering,
and plot daily mean vs median distributions to check skew/outliers.

Assumes:
- data/raw/glucose.csv exists
- Columns: id, study_interval, day_in_study, timestamp, glucose_value
"""

from pathlib import Path
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore

RAW_DIR = Path("data/raw")
INFILE = RAW_DIR / "glucose.csv"
OUT_DIR = Path("results/glucose_checks")
OUT_DIR.mkdir(parents=True, exist_ok=True)

KEYS = ["id", "study_interval", "day_in_study"]
VALUE_COL = "glucose_value"

# Configurable cleaning thresholds (mmol/L)
# Keep this conservative: remove only impossible / clearly invalid values.
MIN_VALID = 0.1 # <= 0 is invalid; keep > 0
MAX_VALID = 30.0 # extremely high; safe bound for "physically implausible" in this context
# Optional: flag (not remove) daily values above this for inspection
FLAG_DAILY_ABOVE = 15.0 # mmol/L; very high for non-diabetic samples but possible in some contexts


def main():
    if not INFILE.exists():
        raise FileNotFoundError(f"Cannot find {INFILE}. Adjust RAW_DIR/INFILE if needed.")

    df = pd.read_csv(INFILE)

    # Basic column checks
    required = set(KEYS + [VALUE_COL])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {INFILE.name}: {sorted(missing)}")

    # Drop obvious non-analytic columns if present
    drop_cols = [c for c in ["is_weekend"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Convert glucose to numeric safely
    df[VALUE_COL] = pd.to_numeric(df[VALUE_COL], errors="coerce")

    # Raw-level invalid filtering BEFORE aggregation
    # Treat non-positive / implausibly high values as missing (do not delete rows; keep timeline integrity).
    invalid_mask = (df[VALUE_COL].isna()) | (df[VALUE_COL] <= MIN_VALID) | (df[VALUE_COL] > MAX_VALID)
    n_invalid = int(invalid_mask.sum())
    df.loc[invalid_mask, VALUE_COL] = pd.NA

    # Quick unit sanity check: raw distribution summary
    raw_desc = df[VALUE_COL].dropna().describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
    print("\n[Raw glucose summary after invalid-value filtering]")
    print(raw_desc)
    print(f"\nInvalid/removed-as-missing raw glucose points: {n_invalid}")

    # Aggregate to daily
    daily = (
        df.groupby(KEYS)[VALUE_COL]
        .agg(glucose_mean="mean", glucose_median="median", glucose_n="count")
        .reset_index()
    )

    # Flag potentially extreme daily values for inspection (do not remove automatically)
    flagged = daily[(daily["glucose_mean"] > FLAG_DAILY_ABOVE) | (daily["glucose_median"] > FLAG_DAILY_ABOVE)]
    print(f"\nFlagged daily rows (mean/median > {FLAG_DAILY_ABOVE} mmol/L): {len(flagged)}")
    if len(flagged) > 0:
        print(flagged.sort_values(["glucose_mean", "glucose_median"], ascending=False).head(20))

    # Save daily summary for later merging
    out_csv = OUT_DIR / "glucose_daily_mean_median.csv"
    daily.to_csv(out_csv, index=False)
    print(f"\nSaved daily glucose summary: {out_csv}")

    # Plot: daily mean vs median distributions
    # Use sane x-limits based on percentiles to avoid one bad point blowing up the axis.
    # (We already filtered > MAX_VALID, but this keeps plots readable even if MAX_VALID is generous.)
    q99 = float(pd.Series(pd.concat([daily["glucose_mean"], daily["glucose_median"]], ignore_index=True)).quantile(0.99))
    x_max = max(10.0, min(MAX_VALID, q99 * 1.25)) # ensure at least up to 10 mmol/L, cap at MAX_VALID

    plt.figure()
    plt.hist(daily["glucose_mean"].dropna(), bins=40, alpha=0.6, label="mean")
    plt.hist(daily["glucose_median"].dropna(), bins=40, alpha=0.6, label="median")
    plt.title("glucose: daily mean vs median")
    plt.xlabel("glucose (mmol/L)")
    plt.ylabel("Count")
    plt.xlim(0, x_max)
    plt.legend()

    out_png = OUT_DIR / "glucose_daily_mean_vs_median.png"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_png}")

    # Optional: quick scatter to see divergence (spike sensitivity)
    plt.figure()
    plt.scatter(daily["glucose_median"], daily["glucose_mean"], s=10, alpha=0.6)
    plt.title("glucose: daily median vs mean (scatter)")
    plt.xlabel("daily median (mmol/L)")
    plt.ylabel("daily mean (mmol/L)")
    # Add diagonal reference
    lim = max(daily["glucose_mean"].max(skipna=True), daily["glucose_median"].max(skipna=True))
    lim = min(float(lim), MAX_VALID)
    plt.plot([0, lim], [0, lim], linestyle="--", linewidth=1)
    plt.xlim(0, lim)
    plt.ylim(0, lim)

    out_png2 = OUT_DIR / "glucose_daily_median_vs_mean_scatter.png"
    plt.savefig(out_png2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_png2}")

    print("\nDone.")


if __name__ == "__main__":
    main()
 


raw = Path("data/raw")
df = pd.read_csv(raw / "glucose.csv")

# Ensure numeric
df["glucose_value"] = pd.to_numeric(df["glucose_value"], errors="coerce")

# Define invalid conditions
invalid_mask = (
    df["glucose_value"].isna() |
    (df["glucose_value"] <= 0) |
    (df["glucose_value"] > 30)
)

invalid = df[invalid_mask]

print("\nTotal raw rows:", len(df))
print("Total invalid raw glucose values:", len(invalid))

if len(invalid) > 0:
    print("\nInvalid examples:")
    print(invalid.sort_values("glucose_value").head(20))
    print(invalid.sort_values("glucose_value", ascending=False).head(20))
else:
    print("\nNo physiologically impossible raw glucose values found.")


df = pd.read_csv("data/raw/glucose.csv")
df["glucose_value"] = pd.to_numeric(df["glucose_value"], errors="coerce")

# 1 Check maximum per ID
max_per_id = df.groupby("id")["glucose_value"].max().sort_values(ascending=False)

print("\nMax glucose per ID:")
print(max_per_id)

# 2 Identify IDs with suspiciously high values (>30)
suspect_ids = max_per_id[max_per_id > 30].index.tolist()

print("\nIDs with max > 30 (likely mg/dL):")
print(suspect_ids)

# 3 Optional: inspect sample rows for those IDs
for sid in suspect_ids:
    print(f"\nSample high values for ID {sid}:")
    print(
        df[df["id"] == sid]
        .sort_values("glucose_value", ascending=False)
        .head(10)
    )
