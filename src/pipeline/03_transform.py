"""
daily_master_v3_transformed.py

Creates a modeling-ready v3 dataset from daily_master_v2 by applying
principled, signal-type–specific transformations:

- Hormones: log(x) for strictly positive concentrations
- Exposure / load variables: log1p(x) for non-negative, burst-like totals
- Physiological state & score variables: left unchanged
- Baseline covariates: left unchanged (but flagged in the log)
- QC / count-like metadata columns can be excluded from transformation by pattern

Also writes a transformation log to CSV and prints a concise summary.

Usage:
    python analyses/preprocessing/daily_master_v3_transformed.py

Inputs:
    data/processed/daily_master_v2.csv

Outputs:
    data/processed/daily_master_v3_transformed.csv
    results/basic_checks_v2/daily_master_v3_transform_log.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import numpy as np  # type: ignore
import pandas as pd  # type: ignore


# Paths
IN_PATH = Path("data/processed/daily_master_v2.csv")
OUT_PATH = Path("data/processed/daily_master_v3_transformed.csv")
LOG_PATH = Path("results/basic_checks_v2/daily_master_v3_transform_log.csv")


# Configuration

# 1) Identifier / timeline columns (never transform)
ID_COLS = {"id", "study_interval", "day_in_study"}

# 2) Static baseline covariates (do not transform; keep for covariates)
BASELINE_COVARS = {
    "age",
    "birth_year",
    "age_of_first_menarche",
    "height_cm",
    "weight_kg",
    "bmi",
}

# 3) Hormones: log(x) (strictly positive expected)
# Adjust names if yours differ (e.g., "e2", "estradiol", "progesterone")
HORMONE_LOG_VARS = {"lh", "estrogen", "pdg"}

# 4) Exposure/load: log1p(x) (non-negative expected)
EXPOSURE_LOG1P_VARS = {
    "azm_cardio_minutes",
    "azm_total_minutes",
    "azm_fat_burn_minutes",
    "actmin_moderately",
    "actmin_very",
    "actmin_lightly",
    "steps_daily",
    "exercise_duration_minutes_sum",
    "hrzones_in_default_zone_1",
    "hrzones_in_default_zone_2",
    "hrzones_in_default_zone_3",
    "hrzones_below_default_zone_1",
    # If you later decide to include workout_count, add here:
    # "exercise_workout_count",
}

# 5) Pattern-based exclusions for QC/count-like variables
# These will NOT be transformed even if numeric and even if they appear in sets above.
EXCLUDE_PATTERNS = (
    "_n",         # e.g., glucose_n, hr_bpm_n
    "_n_points",  # e.g., hrv_n_points
    "_error",     # e.g., rhr_error
)


def is_excluded_by_pattern(col: str) -> bool:
    col_l = col.lower()
    if any(col_l.endswith(p) for p in EXCLUDE_PATTERNS):
        return True
    if "count" in col_l:
        # WARNING: this will also match 'exercise_workout_count' if present.
        # If you want that modeled, remove this line or add an exception.
        return True
    return False


# Helpers
def safe_log(x: pd.Series) -> tuple[pd.Series, dict]:
    x_num = pd.to_numeric(x, errors="coerce")

    n_total = int(x_num.shape[0])
    n_na_before = int(x_num.isna().sum())
    n_nonpos = int((x_num <= 0).sum(skipna=True))

    out = x_num.copy()
    out[out <= 0] = np.nan          # mask first
    out = np.log(out)              # log only valid values

    n_na_after = int(out.isna().sum())

    stats = {
        "n_total": n_total,
        "n_na_before": n_na_before,
        "n_nonpositive_to_nan": n_nonpos,
        "n_na_after": n_na_after,
    }
    return out, stats


def safe_log1p(x: pd.Series) -> tuple[pd.Series, dict]:
    x_num = pd.to_numeric(x, errors="coerce")

    n_total = int(x_num.shape[0])
    n_na_before = int(x_num.isna().sum())
    n_neg = int((x_num < 0).sum(skipna=True))

    out = x_num.copy()
    out[out < 0] = np.nan           # mask first
    out = np.log1p(out)             # log1p only valid values

    n_na_after = int(out.isna().sum())

    stats = {
        "n_total": n_total,
        "n_na_before": n_na_before,
        "n_negative_to_nan": n_neg,
        "n_na_after": n_na_after,
    }
    return out, stats


def summarize_series(x: pd.Series) -> dict:
    """Compact summary of a numeric series for logging."""
    x_num = pd.to_numeric(x, errors="coerce")
    x_nonan = x_num.dropna()
    if x_nonan.empty:
        return {"min": np.nan, "p50": np.nan, "p95": np.nan, "max": np.nan}

    return {
        "min": float(np.nanmin(x_nonan)),
        "p50": float(np.nanmedian(x_nonan)),
        "p95": float(np.nanpercentile(x_nonan, 95)),
        "max": float(np.nanmax(x_nonan)),
    }


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# Main transform
def main() -> int:
    if not IN_PATH.exists():
        print(f"[ERROR] Input file not found: {IN_PATH}", file=sys.stderr)
        return 1

    df = pd.read_csv(IN_PATH)

    # Collect transform log rows
    log_rows: list[dict] = []

    # Work on a copy
    out = df.copy()

    # Identify numeric columns
    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()

    # Iterate columns and apply transformations where appropriate
    for col in numeric_cols:
        action = "none"
        reason = ""
        before = summarize_series(out[col])
        after = {"min": np.nan, "p50": np.nan, "p95": np.nan, "max": np.nan}
        stats: dict = {}

        # Never transform IDs / time indices
        if col in ID_COLS:
            action = "skip"
            reason = "identifier/timeline"
        # Skip pattern-excluded QC/count-like variables
        elif is_excluded_by_pattern(col):
            action = "skip"
            reason = "qc_or_count_like_pattern"
        # Baseline covariates: keep untransformed but note
        elif col in BASELINE_COVARS:
            action = "skip"
            reason = "baseline_covariate"
        # Hormones: log
        elif col in HORMONE_LOG_VARS:
            action = "log"
            reason = "hormone_concentration"
            out[col], stats = safe_log(out[col])
            after = summarize_series(out[col])
        # Exposure/load: log1p
        elif col in EXPOSURE_LOG1P_VARS:
            action = "log1p"
            reason = "exposure_or_load"
            out[col], stats = safe_log1p(out[col])
            after = summarize_series(out[col])
        else:
            action = "none"
            reason = "phys_state_or_score_or_other"

        log_rows.append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "variable": col,
                "action": action,
                "reason": reason,
                "before_min": before["min"],
                "before_p50": before["p50"],
                "before_p95": before["p95"],
                "before_max": before["max"],
                "after_min": after["min"],
                "after_p50": after["p50"],
                "after_p95": after["p95"],
                "after_max": after["max"],
                **stats,
            }
        )

    # Save outputs
    ensure_parent(OUT_PATH)
    ensure_parent(LOG_PATH)

    out.to_csv(OUT_PATH, index=False)
    pd.DataFrame(log_rows).to_csv(LOG_PATH, index=False)

    # Print summary
    log_df = pd.DataFrame(log_rows)

    counts = log_df["action"].value_counts(dropna=False).to_dict()
    print("\n daily_master_v3 created")
    print(f"Input:  {IN_PATH}")
    print(f"Output: {OUT_PATH}")
    print(f"Log:    {LOG_PATH}\n")

    print("Transform counts:")
    for k in ["log", "log1p", "none", "skip"]:
        if k in counts:
            print(f"  {k:5s}: {counts[k]}")

    # Show which variables were transformed
    transformed = log_df[log_df["action"].isin(["log", "log1p"])][["variable", "action", "reason"]]
    if not transformed.empty:
        print("\nTransformed variables:")
        print(transformed.to_string(index=False))

    # Warn if any hormone had nonpositive values
    hormone_warn = log_df[(log_df["action"] == "log") & (log_df.get("n_nonpositive_to_nan", 0) > 0)]
    if not hormone_warn.empty:
        print("\n Warning: Some hormone variables had nonpositive values set to NaN.")
        print(hormone_warn[["variable", "n_nonpositive_to_nan"]].to_string(index=False))

    # Warn if any exposure had negative values
    exposure_warn = log_df[(log_df["action"] == "log1p") & (log_df.get("n_negative_to_nan", 0) > 0)]
    if not exposure_warn.empty:
        print("\n Warning: Some exposure variables had negative values set to NaN.")
        print(exposure_warn[["variable", "n_negative_to_nan"]].to_string(index=False))

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

