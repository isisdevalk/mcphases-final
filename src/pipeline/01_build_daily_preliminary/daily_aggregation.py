"""Core daily-aggregation routines used by the v1 builder.

Defines per-signal aggregators (mean, median, sum, count) and the
join logic that produces the row-per-person-day daily master table."""

import pandas as pd  # type: ignore
from pathlib import Path

raw = Path("data/raw")
processed = Path("data/processed")
processed.mkdir(parents=True, exist_ok=True)

print("\nBuilding daily master dataset (no sleep episode tables)...\n")

# Base timeline
base = pd.read_csv(raw / "hormones_and_selfreport.csv")
daily = base.copy()

keys = ["id", "study_interval", "day_in_study"]

# Helper functions
def dedup_by_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        return df.drop_duplicates(subset=["id", "study_interval", "day_in_study", "timestamp"])
    return df.drop_duplicates()

def clean_hr(df: pd.DataFrame) -> pd.DataFrame:
    if "bpm" in df.columns:
        df["bpm"] = pd.to_numeric(df["bpm"], errors="coerce")
        df.loc[(df["bpm"] < 30) | (df["bpm"] > 230), "bpm"] = pd.NA
    return df

def clean_temp(df: pd.DataFrame) -> pd.DataFrame:
    col = "temperature_diff_from_baseline"
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Conservative physiological bound on ΔT; adjust if you decide otherwise in methods
        df.loc[(df[col] < -15) | (df[col] > 15), col] = pd.NA
    return df

def daily_ready_table(file: str, prefix: str, agg: str = "mean", cleaner=None) -> pd.DataFrame:
    df = pd.read_csv(raw / file)
    df = dedup_by_timestamp(df)

    if cleaner is not None:
        df = cleaner(df)

    drop_cols = [c for c in ["is_weekend", "timestamp"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    if agg == "mean":
        df_daily = df.groupby(keys).mean(numeric_only=True).reset_index()
    elif agg == "first":
        df_daily = df.groupby(keys).first().reset_index()
    else:
        raise ValueError("agg must be 'mean' or 'first'")

    rename = {c: f"{prefix}_{c}" for c in df_daily.columns if c not in keys}
    return df_daily.rename(columns=rename)

def clean_rhr(df: pd.DataFrame) -> pd.DataFrame:
    # Fitbit often uses 0 to mean “no estimate”
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df.loc[df["value"] == 0, "value"] = pd.NA
    if "rhr_value" in df.columns:  # just in case naming differs
        df["rhr_value"] = pd.to_numeric(df["rhr_value"], errors="coerce")
        df.loc[df["rhr_value"] == 0, "rhr_value"] = pd.NA
    return df


def load_height_weight_interval_specific() -> pd.DataFrame:
    """
    Create interval-specific height and weight.
    If study_interval == 2022 → use height_2022 / weight_2022
    If study_interval == 2024 → use height_2024 / weight_2024
    """
    df = pd.read_csv(raw / "height_and_weight.csv")

    for c in df.columns:
        if c != "id":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    rows = []
    for _, row in df.iterrows():
        id_ = row["id"]

        if not pd.isna(row.get("height_2022")) or not pd.isna(row.get("weight_2022")):
            rows.append(
                {"id": id_, "study_interval": 2022, "height_cm": row.get("height_2022"), "weight_kg": row.get("weight_2022")}
            )

        if not pd.isna(row.get("height_2024")) or not pd.isna(row.get("weight_2024")):
            rows.append(
                {"id": id_, "study_interval": 2024, "height_cm": row.get("height_2024"), "weight_kg": row.get("weight_2024")}
            )

    out = pd.DataFrame(rows).drop_duplicates(subset=["id", "study_interval"])
    return out

def load_subject_info() -> pd.DataFrame:
    df = pd.read_csv(raw / "subject-info.csv")
    df = df.drop_duplicates(subset=["id"])

    numeric_cols = ["birth_year", "age_of_first_menarche"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def daily_mean_with_n(file: str, name: str, value_col: str, cleaner=None) -> pd.DataFrame:
    """
    Generic daily aggregation for 'state' variables:
      - deduplicate timestamps
      - optional cleaner (e.g., physiologic bounds)
      - drop non-analytic cols
      - aggregate daily mean + daily count (n)
    Returns: {name}_{value_col}_mean, {name}_{value_col}_n
    """
    df = pd.read_csv(raw / file)
    df = dedup_by_timestamp(df)

    if cleaner is not None:
        df = cleaner(df)

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    drop_cols = [c for c in ["is_weekend", "timestamp"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    out = (
        df.groupby(keys)
          .agg(
              **{
                  f"{name}_{value_col}_mean": (value_col, "mean"),
                  f"{name}_{value_col}_n": (value_col, "count"),
              }
          )
          .reset_index()
    )
    return out

def daily_stress(name: str = "stress") -> pd.DataFrame:
    """
    Stress special handling:
    - deduplicate repeated timestamps
    - keep only valid computed rows (calculation_failed == False)
    - drop NO_DATA rows if status exists
    - aggregate:
        stress_score as mean (+ n)
        points as sum (+ n per points input is typically same as rows kept)
    """
    df = pd.read_csv(raw / "stress_score.csv")
    df = dedup_by_timestamp(df)

    if "calculation_failed" in df.columns:
        df = df[df["calculation_failed"] == False].copy()

    if "status" in df.columns:
        df = df[df["status"] != "NO_DATA"].copy()

    # numeric coercion (safe)
    for c in ["stress_score", "sleep_points", "responsiveness_points", "exertion_points"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    drop_cols = [c for c in ["is_weekend", "timestamp"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    agg_map = {
        "stress_score": ["mean", "count"],
    }
    for c in ["sleep_points", "responsiveness_points", "exertion_points"]:
        if c in df.columns:
            agg_map[c] = ["sum"]

    tmp = df.groupby(keys).agg(agg_map)
    # flatten multiindex columns
    tmp.columns = ["_".join([c[0], c[1]]) for c in tmp.columns.to_flat_index()]
    tmp = tmp.reset_index()

    rename = {}
    if "stress_score_mean" in tmp.columns:
        rename["stress_score_mean"] = f"{name}_stress_score"
    if "stress_score_count" in tmp.columns:
        rename["stress_score_count"] = f"{name}_n"

    for c in ["sleep_points_sum", "responsiveness_points_sum", "exertion_points_sum"]:
        if c in tmp.columns:
            rename[c] = f"{name}_{c.replace('_sum','')}"

    return tmp.rename(columns=rename)

def daily_glucose(name: str = "glucose") -> pd.DataFrame:
    df = pd.read_csv(raw / "glucose.csv")
    df = dedup_by_timestamp(df)

    df["glucose_value"] = pd.to_numeric(df["glucose_value"], errors="coerce")

    # Unit harmonization for IDs 6 and 11 (mg/dL -> mmol/L)
    ids_mgdl = [6, 11]
    df.loc[df["id"].isin(ids_mgdl), "glucose_value"] /= 18.0

    # Remove physiologically impossible values (mmol/L)
    df.loc[(df["glucose_value"] <= 0) | (df["glucose_value"] > 30), "glucose_value"] = pd.NA

    drop_cols = [c for c in ["is_weekend", "timestamp"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    glucose_daily = (
        df.groupby(keys)["glucose_value"]
          .agg(glucose_median="median", glucose_mean="mean", glucose_n="count")
          .reset_index()
          .rename(
              columns={
                  "glucose_median": f"{name}_median",
                  "glucose_mean": f"{name}_mean",
                  "glucose_n": f"{name}_n",
              }
          )
    )
    return glucose_daily

def clean_hormones_zero_as_missing(df: pd.DataFrame, hormone_cols=None) -> pd.DataFrame:
    """
    mcPHASES hormones occasionally use 0.0 as a placeholder for 'not measured'.
    Set hormone values <= 0 to NA so they are treated as missing and won't break log transforms later.
    """
    if hormone_cols is None:
        hormone_cols = ["lh", "estrogen", "pdg"]

    for c in hormone_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.loc[df[c] <= 0, c] = pd.NA

    return df

def daily_hrv(name: str = "hrv") -> pd.DataFrame:
    """
    HRV handling:
    - Deduplicate repeated timestamps
    - Remove RMSSD zeros (invalid)
    - Aggregate:
        HF → median (primary) + mean (sensitivity)
        LF → median (primary) + mean (sensitivity)
        RMSSD → mean (primary) + median (sensitivity)
    - Keep daily n_points (count of valid RMSSD points)
    """
    df = pd.read_csv(raw / "heart_rate_variability_details.csv")
    df = dedup_by_timestamp(df)

    for c in ["rmssd", "high_frequency", "low_frequency"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "rmssd" in df.columns:
        df.loc[df["rmssd"] == 0, "rmssd"] = pd.NA

    drop_cols = [c for c in ["is_weekend", "timestamp"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    hrv_daily = (
        df.groupby(keys)
          .agg(
              rmssd_mean=("rmssd", "mean"),
              rmssd_median=("rmssd", "median"),
              hf_median=("high_frequency", "median"),
              hf_mean=("high_frequency", "mean"),
              lf_median=("low_frequency", "median"),
              lf_mean=("low_frequency", "mean"),
              hrv_n=("rmssd", "count"),
          )
          .reset_index()
          .rename(
              columns={
                  "rmssd_mean": f"{name}_rmssd_mean",
                  "rmssd_median": f"{name}_rmssd_median",
                  "hf_median": f"{name}_hf_median",
                  "hf_mean": f"{name}_hf_mean",
                  "lf_median": f"{name}_lf_median",
                  "lf_mean": f"{name}_lf_mean",
                  "hrv_n": f"{name}_n_points",
              }
          )
    )
    return hrv_daily

def daily_steps(
    name: str = "steps",
    daily_max_valid: int = 200_000,
    per_record_max_valid: int = 5_000,
) -> pd.DataFrame:
    df = pd.read_csv(raw / "steps.csv")
    df = dedup_by_timestamp(df)

    df["steps"] = pd.to_numeric(df["steps"], errors="coerce")
    df.loc[(df["steps"] < 0) | (df["steps"] > per_record_max_valid), "steps"] = pd.NA

    drop_cols = [c for c in ["is_weekend", "timestamp"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    out = (
        df.groupby(keys)["steps"]
          .sum(min_count=1)
          .reset_index(name=f"{name}_daily")
    )

    corrupted = out[out[f"{name}_daily"] > daily_max_valid].copy()
    if len(corrupted) > 0:
        print("\nDeleting corrupted step day(s):")
        print(corrupted)
        out = out[out[f"{name}_daily"] <= daily_max_valid].copy()

    return out

def daily_active_zone_minutes(name: str = "azm") -> pd.DataFrame:
    """
    active_zone_minutes.csv (long format):
    - deduplicate repeated timestamps
    - sum total_minutes per day per heart_zone_id
    - pivot to wide: azm_*_minutes
    - compute total across zones (azm_total_minutes)
    """
    df = pd.read_csv(raw / "active_zone_minutes.csv")
    df = dedup_by_timestamp(df)

    df["total_minutes"] = pd.to_numeric(df["total_minutes"], errors="coerce")

    drop_cols = [c for c in ["is_weekend", "timestamp"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    zone_daily = (
        df.groupby(keys + ["heart_zone_id"])["total_minutes"]
          .sum(min_count=1)
          .reset_index()
    )

    wide = (
        zone_daily.pivot_table(
            index=keys,
            columns="heart_zone_id",
            values="total_minutes",
            aggfunc="sum",
        )
        .reset_index()
    )

    wide.columns = [
        c if c in keys else f"{name}_{str(c).lower()}_minutes"
        for c in wide.columns
    ]

    zone_cols = [c for c in wide.columns if c not in keys]
    wide[f"{name}_total_minutes"] = wide[zone_cols].sum(axis=1, min_count=1)

    return wide

def daily_exercise(name: str = "exercise") -> pd.DataFrame:
    df = pd.read_csv(raw / "exercise.csv")

    df = df.rename(columns={"start_day_in_study": "day_in_study"})

    dedup_cols = [
        c for c in [
            "id",
            "study_interval",
            "day_in_study",
            "start_timestamp",
            "activitytypeid",
            "duration",
        ] if c in df.columns
    ]
    if dedup_cols:
        df = df.drop_duplicates(subset=dedup_cols)
    else:
        df = df.drop_duplicates()

    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")

    # Convert milliseconds to minutes
    df["duration"] = df["duration"] / 60000.0

    daily_ex = (
        df.groupby(keys)
          .agg(
              exercise_workout_count=("duration", "count"),
              exercise_duration_minutes_sum=("duration", "sum"),
          )
          .reset_index()
    )

    return daily_ex

# STATIC variables (merge by id / interval)
print("Processing height_and_weight.csv (interval-specific)")
hw = load_height_weight_interval_specific()
daily = daily.merge(hw, on=["id", "study_interval"], how="left")

print("Processing subject-info.csv (static)")
subj = load_subject_info()
daily = daily.merge(subj, on="id", how="left")

# Derive age per study interval
if "birth_year" in daily.columns:
    daily["age"] = pd.to_numeric(daily["study_interval"], errors="coerce") - pd.to_numeric(daily["birth_year"], errors="coerce")

# BMI (static-ish within interval; stored on daily rows)
if "height_cm" in daily.columns and "weight_kg" in daily.columns:
    daily["bmi"] = daily["weight_kg"] / (daily["height_cm"] / 100.0) ** 2

# DAILY summary variables (already daily)
print("Processing active_minutes.csv (already daily)")
actmin_daily = daily_ready_table("active_minutes.csv", "actmin", agg="mean")
daily = daily.merge(actmin_daily, on=keys, how="left")

print("Processing respiratory_rate_summary.csv (already daily)")
resp_daily = daily_ready_table("respiratory_rate_summary.csv", "resp", agg="mean")
daily = daily.merge(resp_daily, on=keys, how="left")

print("Processing resting_heart_rate.csv (already daily)")
rhr_daily = daily_ready_table("resting_heart_rate.csv", "rhr", agg="mean", cleaner=clean_rhr)
daily = daily.merge(rhr_daily, on=keys, how="left")

print("Processing sleep_score.csv (already daily)")
sleep_score_daily = daily_ready_table("sleep_score.csv", "sleep_score", agg="mean")
daily = daily.merge(sleep_score_daily, on=keys, how="left")

print("Processing time_in_heart_rate_zones.csv (already daily)")
hrzones_daily = daily_ready_table("time_in_heart_rate_zones.csv", "hrzones", agg="mean")
daily = daily.merge(hrzones_daily, on=keys, how="left")

# MEAN wearable variables (with daily n)
print("Processing heart_rate.csv (mean + n + invalid screening)")
hr_daily = daily_mean_with_n("heart_rate.csv", "hr", value_col="bpm", cleaner=clean_hr)
daily = daily.merge(hr_daily, on=keys, how="left")

print("Processing wrist_temperature.csv (mean + n + invalid screening)")
temp_daily = daily_mean_with_n("wrist_temperature.csv", "temp", value_col="temperature_diff_from_baseline", cleaner=clean_temp)
daily = daily.merge(temp_daily, on=keys, how="left")

print("Processing stress_score.csv (filter calculation_failed + mean/sum + n)")
stress_daily = daily_stress("stress")
daily = daily.merge(stress_daily, on=keys, how="left")

# MEDIAN wearable variables
print("Processing glucose.csv (median primary + mean sensitivity + n)")
glucose_daily = daily_glucose("glucose")
daily = daily.merge(glucose_daily, on=keys, how="left")

print("Processing heart_rate_variability_details.csv (HRV daily aggregation + n_points)")
hrv_daily = daily_hrv("hrv")
daily = daily.merge(hrv_daily, on=keys, how="left")

# SUM wearable variables
print("Processing active_zone_minutes.csv (daily sums per zone)")
azm_daily = daily_active_zone_minutes("azm")
daily = daily.merge(azm_daily, on=keys, how="left")

print("Processing steps.csv (daily sum + QC)")
steps_daily = daily_steps("steps", daily_max_valid=200_000)
daily = daily.merge(steps_daily, on=keys, how="left")

print("Processing exercise.csv (workout count + duration)")
exercise_daily = daily_exercise("exercise")
daily = daily.merge(exercise_daily, on=keys, how="left")

# Integrity checks (pre-save)
dup = daily.duplicated(subset=keys).sum()
if dup != 0:
    raise ValueError(f"Duplicate person-days after merges: {dup}. Check merge inflation.")

# Save preliminary daily master
out_path = processed / "daily_master_v1.csv"
daily.to_csv(out_path, index=False)

print("\nDaily master built and saved:")
print(f" - {out_path}")
print("Shape:", daily.shape)
print(daily.head())
print(daily.tail())

# Keep in memory if running interactively:
# python src/daily_aggregation.py
# python -i src/daily_aggregation.py
