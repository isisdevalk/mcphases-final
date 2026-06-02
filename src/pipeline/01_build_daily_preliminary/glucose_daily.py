"""Build daily glucose summaries (mean and median) from the raw glucose CSV."""

import pandas as pd     # type: ignore
from pathlib import Path

raw = Path("data/raw")
df = pd.read_csv(raw / "glucose.csv")

df["glucose_value"] = pd.to_numeric(df["glucose_value"], errors="coerce")

# IDs confirmed to be in mg/dL
ids_mgdl = [6, 11]

# Convert mg/dL → mmol/L (divide by 18)
df.loc[df["id"].isin(ids_mgdl), "glucose_value"] = (
    df.loc[df["id"].isin(ids_mgdl), "glucose_value"] / 18.0
)

print("Conversion complete.")


# After your conversion step:
max_per_id = df.groupby("id")["glucose_value"].max().sort_values(ascending=False)
print(max_per_id.head(15))

# Should now all be in mmol/L range (e.g., max ~ 15–20)
print("Any values > 30 mmol/L left?", (df["glucose_value"] > 30).any())
