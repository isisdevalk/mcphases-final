"""Primary-outcome missingness heatmap, one row per participant (v2 dataset)."""

import matplotlib.pyplot as plt # type: ignore
from matplotlib.colors import LinearSegmentedColormap # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
from pathlib import Path

DATA_PATH = Path("data/processed/daily_master_v2.csv")
OUT_PATH = Path("results/missingness_v2/primary_missingness_heatmap_participant.png")

# Custom color palette (dark wine → cream)
colors = [
 "#5B0E2D",
 "#9E2F50",
 "#D96C7F",
 "#F4B6A6",
 "#FBE8D6",
]
custom_cmap = LinearSegmentedColormap.from_list("missingness_palette", colors, N=256)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Primary variables (same list)
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

print(f"Using {len(primary_vars)} primary variables.")

# Missingness per participant
missing_by_id = df.groupby("id")[primary_vars].apply(lambda x: x.isna().mean())

# Order participants by overall missingness (most missing at top)
missing_by_id["__overall__"] = missing_by_id.mean(axis=1)
missing_by_id = missing_by_id.sort_values("__overall__", ascending=False)
missing_by_id = missing_by_id.drop(columns="__overall__")

data_matrix = missing_by_id.values
ids = missing_by_id.index.tolist()

# Plot
plt.figure(figsize=(16, 10))

im = plt.imshow(
 data_matrix,
 aspect="auto",
 cmap=custom_cmap,
 vmin=0,
 vmax=1
)

plt.xticks(
 ticks=np.arange(len(primary_vars)),
 labels=primary_vars,
 rotation=90
)

plt.yticks(
 ticks=np.arange(len(ids)),
 labels=ids
)

cbar = plt.colorbar(im)
cbar.set_label("Proportion Missing")

plt.title("Missingness by Participant (Primary Variables)")
plt.tight_layout()

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, dpi=300)
plt.close()

print(f"Saved heatmap to: {OUT_PATH}")
print("Done ")