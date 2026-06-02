"""Build a one-row-per-person-day distribution panel from raw signals
and emit per-variable distribution diagnostics (histograms, skewness)."""

import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("results/distributions_all")
OUT.mkdir(parents=True, exist_ok=True)

print("\nBuilding daily dataset for distribution diagnostics...\n")

# Base timeline (1 row per person-day)
daily = pd.read_csv(RAW / "hormones_and_selfreport.csv")
keys = ["id", "study_interval", "day_in_study"]

# Helpers
def add_daily_mean(daily_df: pd.DataFrame, file: str, prefix: str) -> pd.DataFrame:
    """Default: daily mean aggregation."""
    df = pd.read_csv(RAW / file)

    drop_cols = [c for c in ["is_weekend", "timestamp"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df_daily = df.groupby(keys).mean(numeric_only=True).reset_index()

    non_keys = [c for c in df_daily.columns if c not in keys]
    df_daily.rename(columns={c: f"{prefix}_{c}" for c in non_keys}, inplace=True)

    return daily_df.merge(df_daily, on=keys, how="left")


def add_daily_resp(daily_df: pd.DataFrame, prefix: str = "resp") -> pd.DataFrame:
    """
    Respiration cleaning:
    - drop duplicates
    - filter status != NO_DATA (if present)
    - treat 0 as invalid placeholder -> NaN for respiration outputs
    - aggregate mean (daily state)
    """
    df = pd.read_csv(RAW / "respiratory_rate_summary.csv")

    drop_cols = [c for c in ["is_weekend", "timestamp"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df = df.drop_duplicates()

    if "status" in df.columns:
        df = df[df["status"] != "NO_DATA"].copy()

    zero_invalid_cols = [
        "full_sleep_breathing_rate",
        "deep_sleep_breathing_rate",
        "light_sleep_breathing_rate",
        "rem_sleep_breathing_rate",
        "full_sleep_standard_deviation",
        "deep_sleep_standard_deviation",
        "light_sleep_standard_deviation",
        "rem_sleep_standard_deviation",
        "full_sleep_signal_to_noise",
        "deep_sleep_signal_to_noise",
        "light_sleep_signal_to_noise",
        "rem_sleep_signal_to_noise",
    ]

    for c in zero_invalid_cols:
        if c in df.columns:
            df.loc[df[c] == 0, c] = pd.NA

    df_daily = df.groupby(keys).mean(numeric_only=True).reset_index()

    non_keys = [c for c in df_daily.columns if c not in keys]
    df_daily.rename(columns={c: f"{prefix}_{c}" for c in non_keys}, inplace=True)

    return daily_df.merge(df_daily, on=keys, how="left")


def add_daily_stress(daily_df: pd.DataFrame, prefix: str = "stress") -> pd.DataFrame:
    """
    Stress cleaning:
    - only keep rows where calculation_failed == False (valid)
    - stress_score = mean (daily state)
    - subscores points = sum (daily load)
    """
    df = pd.read_csv(RAW / "stress_score.csv")

    drop_cols = [c for c in ["is_weekend", "timestamp"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    if "calculation_failed" in df.columns:
        df = df[df["calculation_failed"] == False].copy()

    df = df.drop_duplicates()

    agg_map = {}
    if "stress_score" in df.columns:
        agg_map["stress_score"] = "mean"

    for c in ["sleep_points", "responsiveness_points", "exertion_points"]:
        if c in df.columns:
            agg_map[c] = "sum"

    df_daily = df.groupby(keys).agg(agg_map).reset_index()

    non_keys = [c for c in df_daily.columns if c not in keys]
    df_daily.rename(columns={c: f"{prefix}_{c}" for c in non_keys}, inplace=True)

    return daily_df.merge(df_daily, on=keys, how="left")


def plot_dist(series: pd.Series, colname: str) -> None:
    data = series.dropna()
    if data.empty:
        print(f"[SKIP] {colname}: no non-missing values")
        return

    plt.figure(figsize=(6, 4))
    sns.histplot(data, kde=True, bins=40)
    plt.title(colname)
    plt.tight_layout()
    plt.savefig(OUT / f"{colname}.png")
    plt.close()


# Add required tables
print("Adding mean-candidate physiology tables...")
daily = add_daily_mean(daily, "heart_rate.csv", "hr")
daily = add_daily_mean(daily, "wrist_temperature.csv", "temp")
daily = add_daily_mean(daily, "estimated_oxygen_variation.csv", "o2var")
daily = add_daily_mean(daily, "altitude.csv", "altitude") # keep zeros (sea level can be 0)

print("Adding HRV + glucose tables...")
daily = add_daily_mean(daily, "heart_rate_variability_details.csv", "hrv")
daily = add_daily_mean(daily, "glucose.csv", "glucose")

print("Adding respiration (clean zeros + NO_DATA)...")
daily = add_daily_resp(daily, "resp")

print("Adding stress (filter calculation_failed)...")
daily = add_daily_stress(daily, "stress")

print("\nDaily dataset built:", daily.shape)

# Build variable list for distributions
# (explicit + auto-include all relevant prefixed columns)
explicit_vars = [
    # Heart rate / phys
    "hr_bpm",
    "temp_temperature",
    "o2var_signal_ratio",
    "altitude_altitude",

    # Glucose + HRV (core ones you care about)
    "glucose_glucose_value",
    "hrv_rmssd",
    "hrv_low_frequency",
    "hrv_high_frequency",

    # Resp breathing rate example (plus variability metrics)
    "resp_full_sleep_breathing_rate",
    "resp_deep_sleep_breathing_rate",
    "resp_light_sleep_breathing_rate",
    "resp_rem_sleep_breathing_rate",
    "resp_full_sleep_standard_deviation",
    "resp_deep_sleep_standard_deviation",
    "resp_light_sleep_standard_deviation",
    "resp_rem_sleep_standard_deviation",
    "resp_full_sleep_signal_to_noise",
    "resp_deep_sleep_signal_to_noise",
    "resp_light_sleep_signal_to_noise",
    "resp_rem_sleep_signal_to_noise",

    # Stress
    "stress_stress_score",
    "stress_sleep_points",
    "stress_responsiveness_points",
    "stress_exertion_points",
]

# Auto-include *everything* from these families, so you don't miss columns
auto_prefixes = ["hr_", "temp_", "o2var_", "altitude_", "glucose_", "hrv_", "resp_", "stress_"]
auto_vars = []
for p in auto_prefixes:
    auto_vars.extend([c for c in daily.columns if c.startswith(p)])

# Deduplicate while preserving order
seen = set()
vars_to_plot = []
for c in explicit_vars + auto_vars:
    if c in daily.columns and c not in seen:
        vars_to_plot.append(c)
    seen.add(c)

print(f"Plotting {len(vars_to_plot)} distributions...")
print("Saving to:", OUT)

# Plot
for col in vars_to_plot:
    plot_dist(daily[col], col)

print("\nDone Distributions saved to:", OUT)
