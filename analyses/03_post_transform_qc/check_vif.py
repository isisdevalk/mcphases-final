"""Variance-inflation-factor (VIF) check on daily_master_v3_transformed.csv."""

import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor # type: ignore

IN_PATH = Path("data/processed/daily_master_v3_transformed.csv")
OUT_PATH = Path("results/collinearity/v3_vif.csv")

KEYS = ["id", "study_interval", "day_in_study"]

EXCLUDE_PATTERNS = ("_n", "_error", "_points")

# mean/median duplicate resolution rules
PREFER = {
    "glucose": "median",
    "hrv_hf": "median",
    "hrv_lf": "median",
    "hrv_rmssd": "mean",
}

# Missingness controls
MAX_MISSING_FRAC = 0.40
MIN_COMPLETE_ROWS = 300


def is_excluded(col: str) -> bool:
    if col in KEYS:
        return True
    cl = col.lower()
    if any(cl.endswith(p) for p in EXCLUDE_PATTERNS):
        return True
    return False


def drop_mean_median_duplicates(cols: list[str]) -> tuple[list[str], list[dict]]:
    cols_set = set(cols)
    dropped_log = []

    def resolve_pair(prefix: str, prefer: str):
        mean_col = f"{prefix}_mean"
        median_col = f"{prefix}_median"

        if mean_col in cols_set and median_col in cols_set:
            if prefer == "median":
                cols_set.remove(mean_col)
                dropped_log.append(
                    {"kept": median_col, "dropped": mean_col}
                )
            elif prefer == "mean":
                cols_set.remove(median_col)
                dropped_log.append(
                    {"kept": mean_col, "dropped": median_col}
                )

    for prefix, prefer in PREFER.items():
        resolve_pair(prefix, prefer)

    kept_cols = [c for c in cols if c in cols_set]
    return kept_cols, dropped_log


def main():
    df = pd.read_csv(IN_PATH)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    predictors = [c for c in numeric_cols if not is_excluded(c)]

    # Drop only mean/median duplicates
    predictors, dropped_log = drop_mean_median_duplicates(predictors)

    if dropped_log:
        print("\nDropped mean/median duplicates:")
        for r in dropped_log:
            print(f"  kept {r['kept']} | dropped {r['dropped']}")
    else:
        print("\nNo mean/median duplicates found.")

    # Filter by missingness
    miss = df[predictors].isna().mean()
    keep = miss[miss <= MAX_MISSING_FRAC].index.tolist()

    if len(keep) < 2:
        print("[WARN] Too few predictors after missingness filtering.")
        return

    X = df[keep].dropna()

    if X.shape[0] < MIN_COMPLETE_ROWS:
        print("[WARN] Too few complete rows for stable VIF.")
        print(f"Complete rows: {X.shape[0]}")
        return

    # Remove constant columns
    nunique = X.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        print("\nDropped constant columns:")
        for c in constant_cols:
            print(" ", c)
        X = X.drop(columns=constant_cols)

    vif_rows = []
    values = X.values

    for i, col in enumerate(X.columns):
        vif_rows.append({
            "variable": col,
            "VIF": float(variance_inflation_factor(values, i))
        })

    vif_df = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    vif_df.to_csv(OUT_PATH, index=False)

    print(f"\nVIF computed on {X.shape[0]} rows and {X.shape[1]} predictors.")
    print(vif_df.head(30).to_string(index=False))
    print("\nSaved to:", OUT_PATH)


if __name__ == "__main__":
    main()