"""Per-participant missingness on daily_master.csv
(post-variable-selection, post-eligibility — output of pipeline stage 04)."""

import pandas as pd # type: ignore
from pathlib import Path

# Load the final selected dataset (output of src/pipeline/04_select_final.py)
DATA_PATH = Path("data/processed/daily_master.csv")
df = pd.read_csv(DATA_PATH)

ID_COL = "id"

# Choose which columns to evaluate missingness on
# (exclude keys/labels)
EXCLUDE_FROM_MISSINGNESS = {ID_COL, "study_interval", "day_in_study", "phase"}
feature_cols = [c for c in df.columns if c not in EXCLUDE_FROM_MISSINGNESS]

# 1) Missingness per variable per participant
#    (matrix: participants x variables)
missing_by_var = (
    df.groupby(ID_COL)[feature_cols]
      .apply(lambda g: g.isna().mean())
      .reset_index()
)

# 2) Overall missingness per participant
#    (mean missingness across selected variables)
participant_missing = (
    missing_by_var.set_index(ID_COL)
                  .mean(axis=1)
                  .reset_index(name="overall_missing_fraction")
)

participant_missing["overall_missing_percent"] = (
    participant_missing["overall_missing_fraction"] * 100
)

# 3) Print summary
print("\n==============================")
print("Participant-Level Missingness (v4)")
print("==============================\n")

print("Top 10 participants with highest overall missingness:\n")
print(
    participant_missing
    .sort_values("overall_missing_percent", ascending=False)
    .head(10)
    .to_string(index=False)
)

print("\nMean missingness across participants:",
      round(participant_missing["overall_missing_percent"].mean(), 2), "%")

# 4) Flag participants above threshold
THRESHOLD = 30
participant_missing["flag_high_missing"] = (
    participant_missing["overall_missing_percent"] > THRESHOLD
)

n_flagged = int(participant_missing["flag_high_missing"].sum())
print(f"\nParticipants with >{THRESHOLD}% missing: {n_flagged}")

# 5) Save results
OUTDIR = Path("results/missingness_v4")
OUTDIR.mkdir(parents=True, exist_ok=True)

participant_missing.to_csv(OUTDIR / "participant_overall_missingness.csv", index=False)
missing_by_var.to_csv(OUTDIR / "participant_missingness_by_variable.csv", index=False)

print("\nSaved results to:", OUTDIR.resolve())

# Which variables are most missing overall?

var_missing = df.isna().mean().sort_values(ascending=False)

print("\nTop 15 most missing variables:\n")
print(var_missing.head(15))

exclude_structural = ["pdg", "bmi", "glucose_median"]

feature_cols_reduced = [c for c in feature_cols if c not in exclude_structural]

missing_reduced = (
    df.groupby("id")[feature_cols_reduced]
      .apply(lambda g: g.isna().mean().mean())
      .reset_index(name="missing_reduced")
)

missing_reduced["missing_reduced_percent"] = missing_reduced["missing_reduced"] * 100

print("\nMean missingness (excluding structural vars):",
      round(missing_reduced["missing_reduced_percent"].mean(), 2), "%")

# Define blocks
self_report_cols = [
    "flow_volume","flow_color","appetite","exerciselevel","headaches",
    "cramps","sorebreasts","fatigue","sleepissue","moodswing",
    "sr_stress","foodcravings","indigestion","bloating"
]

physiology_cols = [
    "hr_bpm_mean","rhr_value","hrv_rmssd_mean",
    "resp_full_sleep_breathing_rate",
    "temp_temperature_diff_from_baseline_mean"
]

df["has_selfreport"] = df[self_report_cols].notna().any(axis=1)
df["has_physiology"] = df[physiology_cols].notna().any(axis=1)
df["has_overlap"] = df["has_selfreport"] & df["has_physiology"]

overlap_summary = df.groupby("id")["has_overlap"].sum().reset_index()
overlap_summary.columns = ["id", "n_overlap_days"]

print(overlap_summary.describe())