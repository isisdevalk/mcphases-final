"""Custom-palette missingness heatmap over participants × variables (v2 dataset)."""

import matplotlib.pyplot as plt # type: ignore
from matplotlib.colors import LinearSegmentedColormap # type: ignore
import pandas as pd # type: ignore
from pathlib import Path

# Custom color palette (dark wine → peach → cream)
colors = [
 "#5B0E2D", # deep wine
 "#9E2F50", # muted raspberry
 "#D96C7F", # soft rose
 "#F4B6A6", # peach
 "#FBE8D6" # light cream
]

custom_cmap = LinearSegmentedColormap.from_list(
 "missingness_palette",
 colors,
 N=256
)
# Load data
DATA_PATH = Path("data/processed/daily_master_v2.csv")
OUT_DIR = Path("results/missingness")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("\nLoading dataset...")
df = pd.read_csv(DATA_PATH)

# Define columns
KEYS = ["id", "study_interval", "day_in_study", "is_weekend", "phase"]
value_cols = [c for c in df.columns if c not in KEYS]

print(f"Computing missingness for {len(value_cols)} variables...\n")

# Compute missingness per participant × variable
missing_matrix = (
 df[value_cols].isna()
 .groupby(df["id"])
 .mean()
)

# Sort participants by overall missingness
missing_matrix["overall_missing"] = missing_matrix.mean(axis=1)
missing_matrix = missing_matrix.sort_values("overall_missing", ascending=False)
missing_matrix = missing_matrix.drop(columns="overall_missing")

# Optionally sort variables by overall missingness
var_order = missing_matrix.mean().sort_values(ascending=False).index
missing_matrix = missing_matrix[var_order]

# Plot heatmap (matplotlib only)
plt.figure(figsize=(20, 10))
plt.imshow(missing_matrix, aspect="auto", cmap=custom_cmap)
plt.colorbar(label="Proportion Missing (0–1)")
plt.clim(0, 1) # ensure scale consistency

plt.xticks(
 ticks=range(len(missing_matrix.columns)),
 labels=missing_matrix.columns,
 rotation=90,
 fontsize=6
)

plt.yticks(
 ticks=range(len(missing_matrix.index)),
 labels=missing_matrix.index
)

plt.title("Participant × Variable Missingness Heatmap")
plt.tight_layout()

plt.savefig(OUT_DIR / "missingness_heatmap.png", dpi=300)
plt.close()

print("\nSaved heatmap to results/missingness/missingness_heatmap.png")
print("\nDone ")