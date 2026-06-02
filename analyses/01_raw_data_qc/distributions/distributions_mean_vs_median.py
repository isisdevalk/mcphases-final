"""Compare daily-mean vs daily-median aggregations side-by-side
to expose skew / outlier sensitivity in raw signals."""

import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("results/distributions_mean_vs_median")
OUT.mkdir(parents=True, exist_ok=True)



keys = ["id", "study_interval", "day_in_study"]

def plot_mean_vs_median(csv_name, value_col, prefix):
    print(f"\nProcessing {csv_name} — {value_col}")

    df = pd.read_csv(RAW / csv_name)

    # Dataset-specific cleaning
    if csv_name == "stress_score.csv":
        if "calculation_failed" in df.columns:
            df = df[df["calculation_failed"] == False].copy()
        df = df[df["status"] != "NO_DATA"]

    df = df.dropna(subset=[value_col])

    # Daily aggregation
    daily_mean = (
        df.groupby(keys)[value_col]
        .mean()
        .reset_index(name="mean")
    )

    daily_median = (
        df.groupby(keys)[value_col]
        .median()
        .reset_index(name="median")
    )

    merged = daily_mean.merge(daily_median, on=keys)

    # Plot
    plt.figure(figsize=(6,4))
    sns.histplot(merged["mean"], kde=True, color="blue", label="mean", bins=40)
    sns.histplot(merged["median"], kde=True, color="orange", label="median", bins=40)
    plt.legend()
    plt.title(f"{prefix}: daily mean vs median")
    plt.tight_layout()
    plt.savefig(OUT / f"{prefix}_mean_vs_median.png")
    plt.close()

    print("Saved:", OUT / f"{prefix}_mean_vs_median.png")


# RUN FOR ALL VARIABLES

plot_mean_vs_median(
    "estimated_oxygen_variation.csv",
    "infrared_to_red_signal_ratio",
    "o2var"
)

plot_mean_vs_median(
    "glucose.csv",
    "glucose_value",
    "glucose"
)

plot_mean_vs_median(
    "heart_rate_variability_details.csv",
    "rmssd",
    "hrv_rmssd"
)

plot_mean_vs_median(
    "heart_rate_variability_details.csv",
    "low_frequency",
    "hrv_lf"
)

plot_mean_vs_median(
    "heart_rate_variability_details.csv",
    "high_frequency",
    "hrv_hf"
)

plot_mean_vs_median(
    "heart_rate.csv",
    "bpm",
    "heart_rate"
)

plot_mean_vs_median(
    "stress_score.csv",
    "stress_score",
    "stress_score"
)

plot_mean_vs_median(
    "wrist_temperature.csv",
    "temperature_diff_from_baseline",
    "temperature"
)

print("\nDone All mean vs median plots saved to:", OUT)
