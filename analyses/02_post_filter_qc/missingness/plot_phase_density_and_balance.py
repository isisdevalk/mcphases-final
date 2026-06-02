"""Density plots of phase coverage and balance per participant (v2 dataset)."""

import pandas as pd # type: ignore
import matplotlib # type: ignore 
matplotlib.use("Agg") # headless backend (no GUI hang)
import matplotlib.pyplot as plt # type: ignore
from matplotlib.colors import LinearSegmentedColormap # type: ignore
from pathlib import Path

# Paths & load
DATA_PATH = Path("data/processed/daily_master_v2.csv")
OUT_DIR = Path("results/missingness/phase_density")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("\nLoading dataset...")
df = pd.read_csv(DATA_PATH)

PHASE_COL = "phase"
KEYS = ["id", "study_interval", "day_in_study", "is_weekend", PHASE_COL]

# keep only rows with phase
df = df.dropna(subset=[PHASE_COL]).copy()

# Ensure phase is treated as categorical with stable ordering if you want
# If you know your canonical order, set it here:
# phase_order = ["menstrual", "follicular", "ovulatory", "luteal"]
# df[PHASE_COL] = pd.Categorical(df[PHASE_COL], categories=phase_order, ordered=True)

value_cols = [c for c in df.columns if c not in KEYS]

print(f"Rows with phase: {len(df)}")
print(f"Participants: {df['id'].nunique()}")
print(f"Variables (non-key): {len(value_cols)}\n")

# Color palette (your gradient)
colors = [
 "#5B0E2D", # deep wine
 "#9E2F50", # muted raspberry
 "#D96C7F", # soft rose
 "#F4B6A6", # peach
 "#FBE8D6" # light cream
]
cmap = LinearSegmentedColormap.from_list("missingness_palette", colors, N=256)

# 1) Overall day counts per phase
counts_days = df.groupby(PHASE_COL).size().sort_values(ascending=False)
counts_days.to_csv(OUT_DIR / "counts_days_per_phase.csv", header=["n_days"])

plt.figure(figsize=(7, 4))
plt.bar(counts_days.index.astype(str), counts_days.values)
plt.ylabel("Number of daily observations")
plt.xlabel("Phase")
plt.title("Data density by phase (days)")
plt.tight_layout()
plt.savefig(OUT_DIR / "phase_density_days.png", dpi=300)
plt.close()

# 2) Participants contributing per phase
# (at least one day in that phase)
counts_participants = df.groupby(PHASE_COL)["id"].nunique().sort_values(ascending=False)
counts_participants.to_csv(OUT_DIR / "counts_participants_per_phase.csv", header=["n_participants"])

plt.figure(figsize=(7, 4))
plt.bar(counts_participants.index.astype(str), counts_participants.values)
plt.ylabel("Number of participants with data")
plt.xlabel("Phase")
plt.title("Participant coverage by phase")
plt.tight_layout()
plt.savefig(OUT_DIR / "phase_coverage_participants.png", dpi=300)
plt.close()

# 3) Phase × variable missingness (descriptive)
# rows: phase, columns: variable, value: proportion missing
missing_by_phase = (
 df[value_cols].isna()
 .groupby(df[PHASE_COL])
 .mean()
)

# Sort variables by overall missingness (descending) to make structure visible
var_order = missing_by_phase.mean(axis=0).sort_values(ascending=False).index
missing_by_phase = missing_by_phase[var_order]

missing_by_phase.to_csv(OUT_DIR / "missingness_by_phase.csv")

# Plot heatmap (phase x variable)
plt.figure(figsize=(18, 4 + 0.4 * len(missing_by_phase.index)))
plt.imshow(missing_by_phase.values, aspect="auto", cmap=cmap) # light=low, dark=high (as defined)
plt.colorbar(label="Proportion missing (0–1)")

plt.yticks(range(len(missing_by_phase.index)), missing_by_phase.index.astype(str))
plt.xticks(range(len(missing_by_phase.columns)), missing_by_phase.columns, rotation=90, fontsize=6)

plt.title("Missingness by phase (descriptive)")
plt.tight_layout()
plt.savefig(OUT_DIR / "missingness_by_phase_heatmap.png", dpi=300)
plt.close()

# Optional: “balance” table for quick reporting
balance = pd.DataFrame({
 "n_days": counts_days,
 "n_participants": counts_participants
}).sort_index()

balance["days_per_participant"] = balance["n_days"] / balance["n_participants"]
balance.to_csv(OUT_DIR / "phase_balance_summary.csv", index=True)

print("Saved figures + tables to:")
print(f" {OUT_DIR}")
print("\nDone ")