"""Per-variable overall missingness on daily_master_v2.csv."""

import pandas as pd # type: ignore
from pathlib import Path

# Paths
DATA_PATH = Path("data/processed/daily_master_v2.csv")
OUT_DIR = Path("results/missingness")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("\nLoading dataset...")
df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(f"Participants: {df['id'].nunique()}")
print(f"Total days: {len(df)}\n")

# 1 Missingness per variable (overall)
print("Calculating overall variable missingness...")

var_missing = (
 df.isna()
 .mean()
 .sort_values(ascending=False)
 .reset_index()
)

var_missing.columns = ["variable", "proportion_missing"]

var_missing["percent_missing"] = var_missing["proportion_missing"] * 100

var_missing.to_csv(OUT_DIR / "missingness_per_variable.csv", index=False)

print("Top 10 variables with highest missingness:")
print(var_missing.head(10), "\n")

# columns to include
KEYS = ["id", "study_interval", "day_in_study"]
value_cols = [c for c in df.columns if c not in KEYS]

# 2 Missingness per participant (overall, across value columns only)
print("Calculating missingness per participant...")

participant_missing = (
 df[value_cols].isna()
 .groupby(df["id"])
 .mean()
 .mean(axis=1)
 .reset_index(name="proportion_missing")
)
participant_missing["percent_missing"] = participant_missing["proportion_missing"] * 100
participant_missing.to_csv(OUT_DIR / "missingness_per_participant.csv", index=False)

print("Participants with highest missingness:")
print(participant_missing.sort_values("percent_missing", ascending=False).head(10), "\n")

# 3 Missingness per participant × variable
print("Calculating missingness per participant × variable...")

participant_var_missing = (
 df[value_cols].isna()
 .groupby(df["id"])
 .mean()
 .reset_index()
)

participant_var_missing.to_csv(OUT_DIR / "missingness_per_participant_variable.csv", index=False)

# 4 Flag high missingness
HIGH_VAR_THRESHOLD = 0.40 # 40%
HIGH_PERSON_THRESHOLD = 0.40 # 40%

high_missing_vars = var_missing[
 var_missing["proportion_missing"] > HIGH_VAR_THRESHOLD
]

high_missing_participants = participant_missing[
 participant_missing["proportion_missing"] > HIGH_PERSON_THRESHOLD
]

high_missing_vars.to_csv(OUT_DIR / "flag_high_missing_variables.csv", index=False)
high_missing_participants.to_csv(OUT_DIR / "flag_high_missing_participants.csv", index=False)

print("Flagged high-missingness variables and participants.\n")

print("Done ")