"""Quality-control checks on the preliminary daily_master before v1 finalization."""

import pandas as pd  # type: ignore
from pathlib import Path

PROCESSED = Path("data/processed")
INFILE = PROCESSED / "daily_master_preliminary.csv"
OUTDIR = PROCESSED / "qc"
OUTDIR.mkdir(parents=True, exist_ok=True)

KEYS = ["id", "study_interval", "day_in_study"]

def main() -> None:
    print("\nQC: daily_master_preliminary.csv\n")
    if not INFILE.exists():
        raise FileNotFoundError(f"Missing file: {INFILE}")

    df = pd.read_csv(INFILE)
    print(f"Loaded: {INFILE}")
    print("Shape:", df.shape)

    # 1) Key integrity
    missing_keys = [k for k in KEYS if k not in df.columns]
    if missing_keys:
        raise ValueError(f"Missing key columns: {missing_keys}")

    dup_n = df.duplicated(subset=KEYS).sum()
    print("\n[1] Duplicate person-days")
    print("Duplicates:", int(dup_n))

    if dup_n > 0:
        dups = df[df.duplicated(subset=KEYS, keep=False)].sort_values(KEYS)
        dups_path = OUTDIR / "duplicate_person_days.csv"
        dups.to_csv(dups_path, index=False)
        print(f"Saved duplicates to: {dups_path}")

    # 2) Completely empty rows
    print("\n[2] Completely empty rows (all NA)")
    all_na_rows = df.isna().all(axis=1).sum()
    print("All-NA rows:", int(all_na_rows))

    if all_na_rows > 0:
        all_na = df[df.isna().all(axis=1)]
        all_na_path = OUTDIR / "all_na_rows.csv"
        all_na.to_csv(all_na_path, index=False)
        print(f"Saved all-NA rows to: {all_na_path}")

    # 3) Basic sanity ranges (soft checks)
    print("\n[3] Sanity range flags (conservative)")
    checks = {}

    if "hr_bpm_mean" in df.columns:
        checks["hr_bpm_mean_outside_30_230"] = df["hr_bpm_mean"].notna() & (
            (df["hr_bpm_mean"] < 30) | (df["hr_bpm_mean"] > 230)
        )

    if "steps_daily" in df.columns:
        checks["steps_daily_negative"] = df["steps_daily"].notna() & (df["steps_daily"] < 0)
        checks["steps_daily_gt_200k"] = df["steps_daily"].notna() & (df["steps_daily"] > 200_000)

    if "glucose_median" in df.columns:
        checks["glucose_median_outside_0_30"] = df["glucose_median"].notna() & (
            (df["glucose_median"] <= 0) | (df["glucose_median"] > 30)
        )

    if "temp_temperature_diff_from_baseline_mean" in df.columns:
        checks["temp_diff_outside_15"] = df["temp_temperature_diff_from_baseline_mean"].notna() & (
            df["temp_temperature_diff_from_baseline_mean"].abs() > 15
        )

    if "bmi" in df.columns:
        checks["bmi_outside_10_70"] = df["bmi"].notna() & ((df["bmi"] < 10) | (df["bmi"] > 70))

    if "sleep_score_overall_score" in df.columns:
        checks["sleep_score_outside_0_100"] = df["sleep_score_overall_score"].notna() & (
            (df["sleep_score_overall_score"] < 0) | (df["sleep_score_overall_score"] > 100)
        )

    # IMPORTANT: after you fix RHR zeros -> NA, this should be clean.
    if "rhr_value" in df.columns:
        checks["rhr_outside_25_120"] = df["rhr_value"].notna() & (
            (df["rhr_value"] < 25) | (df["rhr_value"] > 120)
        )

    flags_report = []
    for name, mask in checks.items():
        n = int(mask.sum())
        flags_report.append((name, n))
        print(f"{name}: {n}")

        if n > 0:
            out_path = OUTDIR / f"{name}.csv"
            df.loc[mask].to_csv(out_path, index=False)
            print(f"  Saved: {out_path}")

    summary = pd.DataFrame(flags_report, columns=["check", "n_flagged"])
    summary_path = OUTDIR / "qc_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved QC summary: {summary_path}")

    # 4) Coverage summaries (n’s)
    print("\n[4] Coverage (n) summaries")
    n_cols = [c for c in df.columns if c.endswith("_n") or c.endswith("_n_points")]
    if n_cols:
        desc = df[n_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
        cov_path = OUTDIR / "coverage_n_describe.csv"
        desc.to_csv(cov_path)
        print(f"Saved coverage describe table: {cov_path}")
        print(desc[["count", "mean", "min", "50%", "95%", "max"]].head(10))
    else:
        print("No n columns found (unexpected).")

    # 5) Zero-as-invalid scan (targeted)
    print("\n[5] Zero-as-invalid scan (targeted)")
    zero_cols = [
        "rhr_value",                      # should NOT be 0 (we saw this issue)
        "glucose_median", "glucose_mean", # should NOT be 0 in mmol/L
        "hrv_rmssd_mean", "hrv_rmssd_median",  # should NOT be 0 if valid
        "sleep_score_overall_score",      # 0 is suspicious
    ]
    zero_report = []
    for col in zero_cols:
        if col in df.columns:
            n0 = int((df[col] == 0).sum())
            zero_report.append((col, n0))
            print(f"{col} == 0: {n0}")
            if n0 > 0:
                out_path = OUTDIR / f"zero_{col}.csv"
                df.loc[df[col] == 0].to_csv(out_path, index=False)
                print(f"  Saved: {out_path}")

    if zero_report:
        zero_df = pd.DataFrame(zero_report, columns=["column", "n_zero"])
        zero_df.to_csv(OUTDIR / "zero_value_summary.csv", index=False)

    # 6) Logical consistency checks (not “wrong”, but worth knowing)
    print("\n[6] Logical consistency checks")

    logical_flags = {}

    # Exercise: workout_count > 0 but duration == 0
    if {"exercise_workout_count", "exercise_duration_minutes_sum"}.issubset(df.columns):
        logical_flags["exercise_count_gt0_but_duration0"] = (
            df["exercise_workout_count"].fillna(0) > 0
        ) & (df["exercise_duration_minutes_sum"].fillna(0) == 0)

    # HRV present but HR missing (rare; could happen, but worth checking)
    if {"hrv_n_points", "hr_bpm_n"}.issubset(df.columns):
        logical_flags["hrv_present_but_hr_missing"] = df["hrv_n_points"].notna() & df["hr_bpm_n"].isna()

    # Glucose median present but glucose_n very small (e.g., <= 5)
    if {"glucose_median", "glucose_n"}.issubset(df.columns):
        logical_flags["glucose_present_but_n_le_5"] = df["glucose_median"].notna() & (df["glucose_n"].fillna(0) <= 5)

    logical_report = []
    for name, mask in logical_flags.items():
        n = int(mask.sum())
        logical_report.append((name, n))
        print(f"{name}: {n}")
        if n > 0:
            out_path = OUTDIR / f"{name}.csv"
            df.loc[mask].to_csv(out_path, index=False)
            print(f"  Saved: {out_path}")

    if logical_report:
        pd.DataFrame(logical_report, columns=["check", "n_flagged"]).to_csv(
            OUTDIR / "logical_checks_summary.csv", index=False
        )

    # 7) Missingness report (top missing columns)
    print("\n[7] Missingness report")
    miss = df.isna().mean().sort_values(ascending=False)
    miss_top = miss.head(20)
    print(miss_top)

    miss_path = OUTDIR / "missingness_fraction.csv"
    miss.to_csv(miss_path, header=["missing_fraction"])
    print(f"Saved full missingness table: {miss_path}")

    print("\nQC complete.\n")


if __name__ == "__main__":
    main()

# python src/qc_daily_master_preliminary.py
