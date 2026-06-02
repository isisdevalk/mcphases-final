"""Compute per-participant missingness over the analysis variables and
apply exclusion thresholds for the modeling sample."""

# analyses/03_post_transform_qc/missingness_per_participant.py
# Computes participant-level missingness, applies exclusion thresholds,
# and runs a simple sensitivity sample definition.
#
# Assumptions:
# - You have a DAILY dataset (one row per id x study_interval x day_in_study)
# - Missingness is computed across a set of "analysis variables" (you choose)
#
# Outputs:
# - results/missingness/participant_missingness.csv
# - results/missingness/missingness_summary.txt
# - results/missingness/missingness_hist.png
#
# Run:
#   python analyses/participant_missingness_summary.py

from pathlib import Path
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

# CONFIG
INFILE = Path("data/processed/daily_master_v3_transformed.csv")  # <-- change if needed
OUTDIR = Path("results/missingness")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Choose the variables that define "complete day" for your primary analyses.
# (Keep this list aligned with your modeling plan.)
PRIMARY_VARS = [
    # symptoms (examples—edit to match your column names)
    "moodswing", "fatigue", "cramps", "flow_volume", "sleepissue",
    # physiology (examples—edit to match your column names)
    "hr_bpm", "temp_temperature_diff_from_baseline",
    "stress_stress_score",
    "hrv_rmssd_mean",
    # context/behavior (examples—edit)
    "steps_daily",
]

# Thresholds
THRESH_PRIMARY_EXCLUDE = 0.50   # exclude >50% missing
THRESH_SENS_EXCLUDE = 0.40      # sensitivity exclude >40%

# Minimum days observed to be included at all (optional)
MIN_DAYS_PER_PERSON = 30

# Keys (must exist in your daily file)
ID_COL = "id"
DAY_KEYS = ["id", "study_interval", "day_in_study"]


def main():
    if not INFILE.exists():
        raise FileNotFoundError(
            f"Could not find {INFILE}. Either save your daily dataset there, "
            f"or update INFILE at the top of this script."
        )

    df = pd.read_csv(INFILE)

    # sanity checks
    for k in DAY_KEYS:
        if k not in df.columns:
            raise ValueError(f"Missing key column '{k}' in {INFILE}")

    # keep only vars that actually exist (prevents crashes if you renamed columns)
    vars_used = [c for c in PRIMARY_VARS if c in df.columns]
    missing_vars = [c for c in PRIMARY_VARS if c not in df.columns]
    if len(vars_used) == 0:
        raise ValueError(
            "None of PRIMARY_VARS exist in the dataset. "
            "Update PRIMARY_VARS to match your daily dataset columns."
        )

    # (optional) ensure one row per day
    # If duplicates exist, you likely want to fix upstream, but we can collapse safely here:
    if df.duplicated(subset=DAY_KEYS).any():
        df = (
            df.sort_values(DAY_KEYS)
              .groupby(DAY_KEYS, as_index=False)
              .first()
        )

    # Create per-day completeness indicator:
    # A day is "complete" if ALL analysis vars are non-missing.
    day_complete = df[vars_used].notna().all(axis=1)

    # Per-person counts
    per_person = (
        df.assign(_complete_day=day_complete)
          .groupby(ID_COL)
          .agg(
              n_days=("day_in_study", "count"),
              n_complete_days=("_complete_day", "sum"),
          )
          .reset_index()
    )

    per_person["proportion_missing"] = 1 - (per_person["n_complete_days"] / per_person["n_days"])
    per_person["percent_missing"] = 100 * per_person["proportion_missing"]

    # Apply minimum-days criterion (optional)
    per_person["fails_min_days"] = per_person["n_days"] < MIN_DAYS_PER_PERSON

    # Primary sample and sensitivity sample flags
    per_person["exclude_primary"] = (per_person["proportion_missing"] > THRESH_PRIMARY_EXCLUDE) | per_person["fails_min_days"]
    per_person["exclude_sensitivity"] = (per_person["proportion_missing"] > THRESH_SENS_EXCLUDE) | per_person["fails_min_days"]

    # Save table
    out_csv = OUTDIR / "participant_missingness.csv"
    per_person.sort_values("proportion_missing").to_csv(out_csv, index=False)

    # Summary stats
    mean_miss = per_person["proportion_missing"].mean()
    median_miss = per_person["proportion_missing"].median()
    n_total = per_person.shape[0]
    n_primary = (~per_person["exclude_primary"]).sum()
    n_sens = (~per_person["exclude_sensitivity"]).sum()

    # Who gets excluded?
    excl_primary_ids = per_person.loc[per_person["exclude_primary"], ID_COL].tolist()
    excl_sens_ids = per_person.loc[per_person["exclude_sensitivity"], ID_COL].tolist()

    # Write a short report
    report = []
    report.append("Participant missingness summary\n")
    report.append(f"Input file: {INFILE}\n")
    report.append(f"Variables used ({len(vars_used)}): {vars_used}\n")
    if missing_vars:
        report.append(f"NOTE: These PRIMARY_VARS were not found and were ignored: {missing_vars}\n")

    report.append(f"\nN participants: {n_total}\n")
    report.append(f"Mean missingness:   {mean_miss:.4f} ({mean_miss*100:.2f}%)\n")
    report.append(f"Median missingness: {median_miss:.4f} ({median_miss*100:.2f}%)\n")

    report.append(f"\nMin-days rule: MIN_DAYS_PER_PERSON = {MIN_DAYS_PER_PERSON}\n")
    report.append(f"Primary exclusion threshold: >{THRESH_PRIMARY_EXCLUDE:.2f} missing\n")
    report.append(f"Sensitivity threshold:       >{THRESH_SENS_EXCLUDE:.2f} missing\n")

    report.append(f"\nPrimary sample N (kept): {n_primary}\n")
    report.append(f"Sensitivity sample N (kept): {n_sens}\n")

    report.append(f"\nExcluded in primary (IDs): {excl_primary_ids}\n")
    report.append(f"Excluded in sensitivity (IDs): {excl_sens_ids}\n")

    out_txt = OUTDIR / "missingness_summary.txt"
    out_txt.write_text("".join(report), encoding="utf-8")

    # Histogram
    plt.figure()
    plt.hist(per_person["percent_missing"], bins=15)
    plt.xlabel("Percent missing days (across selected variables)")
    plt.ylabel("Number of participants")
    plt.title("Participant-level missingness distribution")
    plt.tight_layout()
    out_png = OUTDIR / "missingness_hist.png"
    plt.savefig(out_png, dpi=200)
    plt.close()

    print("\nDone.")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_txt}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()