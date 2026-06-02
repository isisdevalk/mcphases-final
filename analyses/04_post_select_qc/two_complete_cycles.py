"""Identify participants meeting the minimum cycle coverage criterion
(>=3 detected menstrual onsets = >=2 complete cycles)."""

import pandas as pd  # type: ignore
import numpy as np  # type: ignore

# Load final selected dataset
df = pd.read_csv("data/processed/daily_master.csv")

# Settings
FLOW_COL = "flow_volume"

NO_FLOW_VALUES = {
    "Not at all",
    "not at all",
    "None",
    "none",
    "",
}

MIN_CYCLE_STARTS = 3


# Bleeding detection
def is_bleeding(val) -> bool:
    if pd.isna(val):
        return False
    return str(val).strip() not in NO_FLOW_VALUES


# Detect menstrual onsets
df = df.sort_values(["id", "day_in_study"]).copy()

df["is_bleeding"] = df[FLOW_COL].apply(is_bleeding)

df["cycle_start_flag"] = (
    df.groupby("id")["is_bleeding"]
    .transform(lambda x: x & (~x.shift(1, fill_value=False)))
    .astype(int)
)

# Count cycle starts per participant
cycle_summary = (
    df.groupby("id")["cycle_start_flag"]
    .sum()
    .reset_index(name="n_cycle_starts")
)

# Apply >=3 cycle start rule
keep = cycle_summary[
    cycle_summary["n_cycle_starts"] >= MIN_CYCLE_STARTS
]

exclude = cycle_summary[
    cycle_summary["n_cycle_starts"] < MIN_CYCLE_STARTS
]

# Output
print("\nParticipants to KEEP (>=3 menstrual onsets):")
print(sorted(keep["id"].tolist()))

print("\nParticipants to EXCLUDE (<3 menstrual onsets):")
print(exclude.sort_values("n_cycle_starts"))

print("\nNumber kept:", len(keep))
print("Number excluded:", len(exclude))