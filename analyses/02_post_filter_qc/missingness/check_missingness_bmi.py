"""Verify that static covariates (height, weight, BMI) are constant per participant
after the v2 build."""

import pandas as pd # type: ignore

df = pd.read_csv("data/processed/daily_master_v1.csv")

static_vars = ["height_cm", "weight_kg", "bmi"]

for var in static_vars:
    print(f"\nChecking {var}...")

    # number of unique non-missing values per participant
    variability = (
        df.groupby("id")[var]
          .nunique(dropna=True)
    )

    problematic = variability[variability > 1]

    print(f"Participants with >1 unique value: {len(problematic)}")

    if len(problematic) > 0:
        print(problematic.head())