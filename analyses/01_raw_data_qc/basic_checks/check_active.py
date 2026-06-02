"""Inspect distinct heart_zone_id values in the raw active_zone_minutes CSV."""

import pandas as pd # type: ignore

df = pd.read_csv("data/raw/active_zone_minutes.csv")

print("Unique heart_zone_id values:")
print(df["heart_zone_id"].value_counts())

print("\nDistinct values (sorted):")
print(sorted(df["heart_zone_id"].unique()))
