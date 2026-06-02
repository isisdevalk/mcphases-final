"""Primary-outcome missingness heatmap aggregated by menstrual-cycle phase (v2 dataset)."""

import matplotlib.pyplot as plt # type: ignore
from matplotlib.colors import LinearSegmentedColormap # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
from pathlib import Path

DATA_PATH = Path("data/processed/daily_master_v2.csv")
OUT_PATH = Path("results/missingness_v2/primary_missingness_heatmap_phases.png")

# Custom color palette
colors = [
 "#5B0E2D", # deep wine (high missing)
 "#9E2F50",
 "#D96C7F",
 "#F4B6A6",
 "#FBE8D6" # cream (low missing)
]

custom_cmap = LinearSegmentedColormap.from_list(
 "missingness_palette",
 colors,
 N=256
)

# Load data
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Define primary variables
primary_vars = [
 "hr_bpm_mean",
 "rhr_value",
 "stress_stress_score",
 "temp_temperature_diff_from_baseline_mean",
 "glucose_median",
 "hrv_rmssd_mean",
 "hrv_hf_median",
 "hrv_lf_median",
 "sleep_score_overall_score",
 "sleep_score_duration_score",
 "sleep_score_deep_sleep_in_minutes",
 "steps_daily",
 "azm_total_minutes",
 "actmin_moderately",
 "actmin_very",
 "moodswing",
 "fatigue",
 "cramps",
 "bloating",
 "sleepissue",
 "stress",
]

primary_vars = [v for v in primary_vars if v in df.columns]

# Compute phase-wise missingness
heatmap_data = (
 df.groupby("phase")[primary_vars]
 .apply(lambda x: x.isna().mean())
)

# Ensure consistent phase order if needed
phase_order = sorted(heatmap_data.index)
heatmap_data = heatmap_data.loc[phase_order]

# Convert to numpy
data_matrix = heatmap_data.values

# Plot
plt.figure(figsize=(16, 6))

im = plt.imshow(
 data_matrix,
 aspect="auto",
 cmap=custom_cmap,
 vmin=0,
 vmax=1
)

# Axis formatting
plt.xticks(
 ticks=np.arange(len(primary_vars)),
 labels=primary_vars,
 rotation=90
)

plt.yticks(
 ticks=np.arange(len(phase_order)),
 labels=phase_order
)

cbar = plt.colorbar(im)
cbar.set_label("Proportion Missing")

plt.title("Phase-Stratified Missingness (Primary Variables)")
plt.tight_layout()

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, dpi=300)
plt.close()

print(f"Saved heatmap to: {OUT_PATH}")
print("Done ")