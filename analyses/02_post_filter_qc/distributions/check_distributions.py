"""Per-variable distribution summary (skewness, kurtosis) on daily_master_v2.csv."""

import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from pathlib import Path
from scipy.stats import skew  # type: ignore

DATA_PATH = Path("data/processed/daily_master_v2.csv")
OUT_PATH = Path("results/distribution_summary.csv")

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Only numeric variables
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Remove identifiers
numeric_cols = [c for c in numeric_cols if c not in ["id", "study_interval", "day_in_study"]]

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Remove identifiers
exclude = {"id", "study_interval", "day_in_study"}

# Remove count-like variables
exclude.update([c for c in numeric_cols if c.endswith("_n")])
exclude.update([c for c in numeric_cols if c.endswith("_n_points")])
exclude.update([c for c in numeric_cols if c.endswith("_error")])
exclude.update([c for c in numeric_cols if "count" in c.lower()])

numeric_cols = [c for c in numeric_cols if c not in exclude]

results = []

for col in numeric_cols:
    series = df[col].dropna()

    if len(series) == 0:
        continue

    mean = series.mean()
    std = series.std()
    min_val = series.min()
    max_val = series.max()
    zero_prop = (series == 0).mean()
    missing_prop = df[col].isna().mean()
    skewness = skew(series)

    # extreme outliers (>3 SD)
    if std > 0:
        extreme_prop = ((series > mean + 3*std) | (series < mean - 3*std)).mean()
    else:
        extreme_prop = 0

    results.append({
        "variable": col,
        "mean": mean,
        "std": std,
        "min": min_val,
        "max": max_val,
        "percent_missing": missing_prop * 100,
        "percent_zero": zero_prop * 100,
        "skewness": skewness,
        "percent_extreme_3sd": extreme_prop * 100
    })
    
    

summary = pd.DataFrame(results).sort_values("skewness", key=abs, ascending=False)

summary.to_csv(OUT_PATH, index=False)

print("\nSaved distribution summary to:", OUT_PATH)
print("\nTop 10 most skewed variables:")
print(summary.head(10))