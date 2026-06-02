"""Pairwise Pearson correlations on daily_master_v3_transformed.csv;
emit pairs above THRESH for collinearity review."""

import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from pathlib import Path

IN_PATH = Path("data/processed/daily_master_v3_transformed.csv")
OUT_PATH = Path("results/collinearity/v3_high_correlations.csv")

KEYS = ["id", "study_interval", "day_in_study"]

EXCLUDE_PATTERNS = ("_n", "_error", "_points")  # keep workout_count if you want; don't exclude "count" here

# mean/median duplicate resolution rules
# keep one, drop the other if both present
PREFER = {
    # glucose: primary = median
    "glucose": "median",
    # HRV: HF/LF primary = median; RMSSD primary = mean
    "hrv_hf": "median",
    "hrv_lf": "median",
    "hrv_rmssd": "mean",
}

THRESH = 0.80
MIN_PERIODS = 200


def is_excluded(col: str) -> bool:
    if col in KEYS:
        return True
    cl = col.lower()
    if any(cl.endswith(p) for p in EXCLUDE_PATTERNS):
        return True
    return False


def drop_mean_median_duplicates(cols: list[str]) -> tuple[list[str], list[dict]]:
    """
    Drops duplicate mean/median versions based on PREFER rules.
    Returns kept_cols and a log of dropped columns.
    """
    cols_set = set(cols)
    dropped_log = []

    def resolve_pair(prefix: str, prefer: str):
        mean_col = f"{prefix}_mean"
        median_col = f"{prefix}_median"

        if mean_col in cols_set and median_col in cols_set:
            if prefer == "median":
                cols_set.remove(mean_col)
                dropped_log.append(
                    {"kept": median_col, "dropped": mean_col, "reason": "drop_mean_keep_median"}
                )
            elif prefer == "mean":
                cols_set.remove(median_col)
                dropped_log.append(
                    {"kept": mean_col, "dropped": median_col, "reason": "drop_median_keep_mean"}
                )

    for prefix, prefer in PREFER.items():
        resolve_pair(prefix, prefer)

    kept_cols = [c for c in cols if c in cols_set]  # preserve original order
    return kept_cols, dropped_log


def main():
    df = pd.read_csv(IN_PATH)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    predictors = [c for c in numeric_cols if not is_excluded(c)]

    if len(predictors) < 2:
        print("[WARN] Not enough predictors to compute correlations.")
        return

    # Drop mean/median duplicates (only where both exist)
    predictors, dropped_log = drop_mean_median_duplicates(predictors)

    if dropped_log:
        print("\nDropped mean/median duplicates:")
        for r in dropped_log:
            print(f"  kept {r['kept']} | dropped {r['dropped']} ({r['reason']})")
    else:
        print("\nNo mean/median duplicates found to drop.")

    corr = df[predictors].corr(method="pearson", min_periods=MIN_PERIODS)

    high_corr = []
    cols = corr.columns.tolist()

    for i in range(len(cols)):
        for j in range(i):
            val = corr.iloc[i, j]
            if pd.isna(val):
                continue
            if abs(val) >= THRESH:
                high_corr.append({"var1": cols[i], "var2": cols[j], "correlation": float(val)})

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not high_corr:
        pd.DataFrame(columns=["var1", "var2", "correlation"]).to_csv(OUT_PATH, index=False)
        print(f"\nNo high correlations found (|r| >= {THRESH}).")
        print("Saved empty table to:", OUT_PATH)
        return

    out = pd.DataFrame(high_corr)
    out["abs_corr"] = out["correlation"].abs()
    out = out.sort_values("abs_corr", ascending=False).drop(columns=["abs_corr"])

    out.to_csv(OUT_PATH, index=False)

    print(f"\nHigh correlations (|r| >= {THRESH}):")
    print(out.to_string(index=False))
    print("\nSaved to:", OUT_PATH)


if __name__ == "__main__":
    main()