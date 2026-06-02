"""Pipeline stage 04: select final variables and apply eligibility.

Creates daily_master.csv from the v3-transformed master by keeping only the
final selected variables, applying raw eligibility (>= 3 menses starts), and
reporting included/excluded vars.

Inputs:
- data/processed/daily_master_v3_transformed.csv

Outputs:
- data/processed/daily_master.csv          (final modeling-ready dataset)
- results/variable_selection/*.csv         (optional reports)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

# Paths
IN_PATH = Path("data/processed/daily_master_v3_transformed.csv")
OUT_PATH = Path("data/processed/daily_master.csv")

EXPORT = True
EXPORT_DIR = Path("results/variable_selection")

# Inclusion threshold
MIN_MENSES_STARTS = 3

# Cycle start / bleeding logic
NO_FLOW_VALUES = {
    "Not at all",
    "not at all",
    "None",
    "none",
    "",
}

# Columns to ignore as features
KEY_COLS = {"id", "study_interval", "day_in_study"}
NON_FEATURE_COLS = {
    "timestamp",
    "date",
    "datetime",
    "cycle_day",
    "phase",
    "status",
    "calculation_failed",
    "cycle_start_flag",
}
IGNORE_COLS = KEY_COLS | NON_FEATURE_COLS

# Always keep structural/meta columns
ALWAYS_KEEP_META = [
    "date",
    "timestamp",
    "datetime",
    "cycle_day",
    "phase",
    "status",
    "calculation_failed",
    "cycle_start_flag",
]

# INCLUDED variables
INCLUDED_GROUPS: dict[str, list[str]] = {
    "Stable Covariates": [
        "age",
        "bmi",
        "age_of_first_menarche",
    ],
    "Self-report symptoms": [
        "flow_volume",
        "flow_color",
        "appetite",
        "exerciselevel",
        "headaches",
        "cramps",
        "sorebreasts",
        "fatigue",
        "sleepissue",
        "moodswing",
        "stress",
        "foodcravings",
        "indigestion",
        "bloating",
    ],
    "Hormones": [
        "estrogen",
        "lh",
        "pdg",
    ],
    "Core Physiology": [
        "hr_bpm_mean",
        "rhr_value",
        "hrv_rmssd_mean",
        "resp_full_sleep_breathing_rate",
        "resp_full_sleep_standard_deviation",
        "temp_temperature_diff_from_baseline_mean",
        "glucose_median",
    ],
    "Sleep": [
        "sleep_score_overall_score",
        "sleep_score_deep_sleep_in_minutes",
        "sleep_score_restlessness",
    ],
    "Activity / Behavioral Load": [
        "steps_daily",
        "azm_total_minutes",
    ],
    "Context": [
        "stress_stress_score",
    ],
}

OPTIONAL_VARS = {"sleep_score_rem_sleep_in_minutes"}

INCLUDED_VARS = [v for group in INCLUDED_GROUPS.values() for v in group]
ALWAYS_KEEP_KEYS = ["id", "study_interval", "day_in_study"]

# Ordinal encoding
ORDINAL_SELFREPORT_COLS = [
    "flow_volume",
    "appetite",
    "exerciselevel",
    "headaches",
    "cramps",
    "sorebreasts",
    "fatigue",
    "sleepissue",
    "moodswing",
    "stress",
    "foodcravings",
    "indigestion",
    "bloating",
]

DEFAULT_ORD_MAP = {
    "not at all": 0, "none": 0,
    "very low/little": 1, "very low": 1, "little": 1,
    "somewhat low": 2, "low": 2, "somewhat light": 2,
    "moderate": 3, "medium": 3, "somewhat heavy": 3,
    "high": 4, "heavy": 4,
    "very high": 5, "very heavy": 5, "severe": 5,
    "light": 1,
    "spotting / very light": 0,
    "spotting": 0,
    "very light": 0,
}

# QC
RESP_RATE_MIN = 8.0
RESP_RATE_MAX = 30.0


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def is_bleeding_from_flow(x: object) -> bool:
    if pd.isna(x):
        return False
    return str(x).strip() not in NO_FLOW_VALUES


def add_cycle_start_flag_from_flow(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["id", "study_interval", "day_in_study"]).copy()
    df["_bleed"] = df["flow_volume"].map(is_bleeding_from_flow)

    prev = df.groupby(["id", "study_interval"])["_bleed"].shift(1).eq(True)
    df["cycle_start_flag"] = (df["_bleed"] & ~prev).astype(int)

    return df.drop(columns="_bleed")


def filter_participants(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = df.groupby("id")["cycle_start_flag"].sum().reset_index(name="n_menses_starts")
    keep_ids = counts.loc[counts["n_menses_starts"] >= MIN_MENSES_STARTS, "id"]
    return df[df["id"].isin(keep_ids)].copy(), counts


def apply_physiology_qc(df: pd.DataFrame):
    df = df.copy()
    qc = {}

    if "resp_full_sleep_breathing_rate" in df.columns:
        col = "resp_full_sleep_breathing_rate"
        df[col] = pd.to_numeric(df[col], errors="coerce")

        mask = (df[col] == 0) | (df[col] < RESP_RATE_MIN) | (df[col] > RESP_RATE_MAX)
        qc["resp_invalid"] = int(mask.sum())

        df.loc[mask, col] = np.nan

    return df, qc


def encode_ordinal(df: pd.DataFrame, col: str):
    return df[col].map(lambda x: DEFAULT_ORD_MAP.get(str(x).lower(), np.nan))

def print_variable_counts(df: pd.DataFrame, title: str = "VARIABLE VALUE COUNTS") -> None:
    print_header(title)

    rows = []
    for col in df.columns:
        n_nonmissing = int(df[col].notna().sum())
        n_missing = int(df[col].isna().sum())
        missing_pct = 100 * n_missing / len(df) if len(df) > 0 else np.nan

        rows.append({
            "variable": col,
            "n_nonmissing": n_nonmissing,
            "n_missing": n_missing,
            "missing_pct": round(missing_pct, 2),
        })

    counts = pd.DataFrame(rows).sort_values("n_nonmissing", ascending=True)

    print(counts.to_string(index=False)) 
    
def main():
    df = pd.read_csv(IN_PATH)

    # Keep relevant columns
    df = df.copy()

    # Rename
    df = df.rename(columns={
        "stress": "sr_stress",
        "stress_stress_score": "wear_stress_score",
    })

    # Sort
    df = df.sort_values(["id", "study_interval", "day_in_study"])

    # Ensure cycle_start_flag
    if "cycle_start_flag" not in df.columns:
        df = add_cycle_start_flag_from_flow(df)

    # Apply 3 menses start filter
    print_header("RAW ELIGIBILITY — ≥3 MENSES STARTS")

    before = df["id"].nunique()
    df, counts = filter_participants(df)

    print(f"Participants kept: {df['id'].nunique()} / {before}")

    # QC
    df, qc = apply_physiology_qc(df)

    # Ordinal encoding
    for col in ORDINAL_SELFREPORT_COLS:
        col2 = "sr_stress" if col == "stress" else col
        if col2 in df.columns:
            df[f"{col2}__ord"] = encode_ordinal(df, col2)


    # Print variable counts
    print_variable_counts(df, "FINAL VARIABLE COUNTS BEFORE SAVE")
        # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print_header("DONE")
    print(f"Final rows: {len(df)}")
    print(f"Participants: {df['id'].nunique()}")

if __name__ == "__main__":
    main()