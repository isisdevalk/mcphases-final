"""Spot-check non-positive hormone values that would break log-transformation."""

import pandas as pd # type: ignore

df = pd.read_csv("data/processed/daily_master_v2.csv")

for col in ["lh", "estrogen"]:
    bad = df[df[col].notna() & (df[col] <= 0)][["id", "study_interval", "day_in_study", col]]
    print("\n", col, "nonpositive rows:\n", bad)