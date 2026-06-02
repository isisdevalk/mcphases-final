"""Range and duplicate checks on raw state variables (heart rate, wrist temp,
estimated O2 variation)."""

import pandas as pd   # type: ignore
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("results/qc_state_vars")
OUT.mkdir(parents=True, exist_ok=True)

KEYS = ["id", "study_interval", "day_in_study"]
TIMEKEY = KEYS + ["timestamp"]  # only if timestamp exists

def qc_duplicates(df: pd.DataFrame, label: str):
    dup_full = df.duplicated().sum()
    dup_time = df.duplicated(subset=[c for c in TIMEKEY if c in df.columns]).sum() if "timestamp" in df.columns else None
    print(f"\n[{label}] duplicates")
    print("  exact duplicate rows:", int(dup_full))
    if dup_time is not None:
        print("  duplicate timestamp rows:", int(dup_time))
    return int(dup_full), (int(dup_time) if dup_time is not None else None)

def invalidate_range(df: pd.DataFrame, col: str, low=None, high=None):
    """Set values outside [low, high] to NA (boundaries optional). Returns invalidated count."""
    if col not in df.columns:
        return 0
    df[col] = pd.to_numeric(df[col], errors="coerce")
    before = df[col].notna().sum()

    mask = df[col].isna()
    if low is not None:
        mask |= (df[col] < low)
    if high is not None:
        mask |= (df[col] > high)

    df.loc[mask, col] = pd.NA
    after = df[col].notna().sum()
    return int(before - after)

def invalidate_leq(df: pd.DataFrame, col: str, threshold: float = 0):
    """Set values <= threshold to NA. Returns invalidated count."""
    if col not in df.columns:
        return 0
    df[col] = pd.to_numeric(df[col], errors="coerce")
    before = df[col].notna().sum()
    df.loc[df[col].isna() | (df[col] <= threshold), col] = pd.NA
    after = df[col].notna().sum()
    return int(before - after)

def main():
    report_rows = []

    # 1) HEART RATE
    hr_file = "heart_rate.csv"
    hr_col = "bpm"
    df = pd.read_csv(RAW / hr_file)
    qc_duplicates(df, hr_file)

    invalidated = invalidate_range(df, hr_col, low=30, high=230)
    print(f"[{hr_file}] invalidated {invalidated} values in '{hr_col}' (<30 or >230)")

    # Save example invalids (optional)
    inv = df[(pd.to_numeric(df[hr_col], errors="coerce").isna())]  # after invalidation many are NA
    inv.head(200).to_csv(OUT / "heart_rate_invalid_examples.csv", index=False)

    report_rows.append({
        "file": hr_file, "variable": hr_col,
        "rule": "set NA if <30 or >230",
        "invalidated_n": invalidated
    })

    # 2) WRIST TEMPERATURE
    temp_file = "wrist_temperature.csv"
    df = pd.read_csv(RAW / temp_file)
    qc_duplicates(df, temp_file)

    # choose column automatically
    temp_candidates = ["temperature_diff_from_baseline", "temperature"]
    temp_col = next((c for c in temp_candidates if c in df.columns), None)

    invalidated = 0
    rule = "N/A (temp column not found)"
    if temp_col is not None:
        # If it's diff-from-baseline (likely), use conservative bounds [-10, +10] °C.
        # If it's absolute temperature, you might prefer [25, 45] °C.
        if temp_col == "temperature_diff_from_baseline":
            invalidated = invalidate_range(df, temp_col, low=-10, high=10)
            rule = "set NA if diff < -10 or > 10 °C"
        else:
            invalidated = invalidate_range(df, temp_col, low=25, high=45)
            rule = "set NA if temp < 25 or > 45 °C"

        print(f"[{temp_file}] invalidated {invalidated} values in '{temp_col}' ({rule})")
        df[df[temp_col].isna()].head(200).to_csv(OUT / "wrist_temp_invalid_examples.csv", index=False)

    report_rows.append({
        "file": temp_file, "variable": temp_col,
        "rule": rule,
        "invalidated_n": invalidated
    })

    # 3) STRESS SCORE (already special, but QC-only here)
    stress_file = "stress_score.csv"
    df = pd.read_csv(RAW / stress_file)
    qc_duplicates(df, stress_file)

    # mimic your logic: drop calculation_failed and NO_DATA if present
    before = len(df)
    if "calculation_failed" in df.columns:
        df = df[df["calculation_failed"] == False].copy()
    if "status" in df.columns:
        df = df[df["status"] != "NO_DATA"].copy()
    after = len(df)
    dropped = int(before - after)

    print(f"[{stress_file}] rows removed by flags (calculation_failed/NO_DATA): {dropped}")

    report_rows.append({
        "file": stress_file, "variable": "stress_score",
        "rule": "drop rows where calculation_failed==True and/or status==NO_DATA",
        "invalidated_n": dropped
    })

    # Save QC summary
    report = pd.DataFrame(report_rows)
    out_csv = OUT / "qc_invalid_screening_summary.csv"
    report.to_csv(out_csv, index=False)

    print("\nSaved QC summary to:", out_csv)
    print("Saved example files (first 200 rows each) to:", OUT)
    print("\nDone.")

if __name__ == "__main__":
    main()

