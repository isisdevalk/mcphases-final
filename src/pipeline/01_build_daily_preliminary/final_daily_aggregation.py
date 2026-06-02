"""Pipeline stage 01: build daily_master_v1.csv from the raw mcPHASES CSVs."""

import pandas as pd # type: ignore
from pathlib import Path

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

WEARABLES_IN = PROCESSED / "preliminary_daily_data" / "daily_master_preliminary.csv"
BASE_IN = RAW / "hormones_and_selfreport.csv"
OUT = PROCESSED / "daily_master_v1.csv"

KEYS = ["id", "study_interval", "day_in_study"]

def clean_hormones_zero_as_missing(df: pd.DataFrame) -> pd.DataFrame:
 hormone_cols = ["lh", "estrogen", "pdg"]
 for c in hormone_cols:
 if c in df.columns:
 df[c] = pd.to_numeric(df[c], errors="coerce")
 df.loc[df[c] <= 0, c] = pd.NA
 return df

def main():
 print("\nMerging hormones + self-report into wearables (without duplicate columns)...\n")

 wear = pd.read_csv(WEARABLES_IN)
 if not set(KEYS).issubset(wear.columns):
 raise ValueError(f"WEARABLES_IN must contain keys: {KEYS}")

 base = pd.read_csv(BASE_IN).drop_duplicates(subset=KEYS).copy()
 base = clean_hormones_zero_as_missing(base)

 # Drop any base columns from wearables to prevent _x/_y
 overlap = (set(wear.columns) & set(base.columns)) - set(KEYS)
 if overlap:
 print(f"Found {len(overlap)} overlapping columns in WEARABLES_IN. Dropping them from wearables:")
 print(sorted(list(overlap))[:30], "..." if len(overlap) > 30 else "")
 wear = wear.drop(columns=sorted(overlap))

 out = base.merge(wear, on=KEYS, how="left")

 dup = out.duplicated(subset=KEYS).sum()
 if dup != 0:
 raise ValueError(f"Duplicate person-days after merge: {dup}")

 out.to_csv(OUT, index=False)
 print("\n Saved merged daily master:")
 print(f" - {OUT}")
 print("Shape:", out.shape)

if __name__ == "__main__":
 main()