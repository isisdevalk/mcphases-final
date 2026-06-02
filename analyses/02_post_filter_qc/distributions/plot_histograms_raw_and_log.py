"""Side-by-side raw vs log-transformed histograms on daily_master_v2.csv."""

import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from scipy.stats import skew # type: ignore
from pathlib import Path

DATA_PATH = Path("data/processed/daily_master_v2.csv")
OUT_DIR = Path("results/histograms")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Select numeric variables
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Exclude identifiers and quality variables
exclude = {"id", "study_interval", "day_in_study"}
exclude.update([c for c in numeric_cols if c.endswith("_n")])
exclude.update([c for c in numeric_cols if "error" in c.lower()])

plot_vars = [c for c in numeric_cols if c not in exclude]

print(f"Plotting {len(plot_vars)} variables...")

transform_candidates = []

for col in plot_vars:
 series = df[col].dropna()

 if len(series) < 20:
 continue

 # Raw histogram
 plt.figure(figsize=(6,4))
 plt.hist(series, bins=40)
 plt.title(f"{col} (raw)")
 plt.tight_layout()
 plt.savefig(OUT_DIR / f"{col}_raw.png")
 plt.close()

 # Check skew
 s = skew(series)
 if abs(s) > 2:
 transform_candidates.append((col, s))

 # log1p but robust to sentinel values like -1
 x = np.log1p(series)

 # remove non-finite values (-inf, inf, nan)
 x = x.replace([np.inf, -np.inf], np.nan).dropna()

 # only plot if enough data left
 if len(x) >= 20:
 plt.figure(figsize=(6,4))
 plt.hist(x, bins=40)
 plt.title(f"{col} (log1p, finite only)")
 plt.tight_layout()
 plt.savefig(OUT_DIR / f"{col}_log.png")
 plt.close()

print("\nVariables with |skew| > 2:")
for var, s in transform_candidates:
 print(f"{var}: skew = {s:.2f}")

print("\nDone ")