"""Quick scan of every raw CSV to report whether it carries a `day_in_study` column."""

# analyses/check_timeline_columns.py

import pandas as pd # type: ignore
from pathlib import Path

raw_dir = Path("data/raw")

print("Checking for 'day_in_study' in data files:\n")

for csv_file in sorted(raw_dir.glob("*.csv")):
    try:
        df = pd.read_csv(csv_file, nrows=1)  # only read header + 1 row
        if "day_in_study" in df.columns:
            print(f"[OK]    {csv_file.name}")
        else:
            print(f"[MISSING] {csv_file.name} → 'day_in_study' not in data file")
    except Exception as e:
        print(f"[ERROR] {csv_file.name}: {e}")


print("\nFiles missing 'day_in_study' and their variables:\n")

for csv_file in sorted(raw_dir.glob("*.csv")):
    df = pd.read_csv(csv_file, nrows=5)

    if "day_in_study" not in df.columns:
        print(f"\n--- {csv_file.name} ---")
        print(list(df.columns))
        
        
        
""" sleep.csv & computed_temperature.csv: Sleep data are episode-based and indexed by start and end days; for daily analyses, 
I should align sleep to the day it ends using sleep_end_day_in_study, typically keeping only the main sleep episode."""

""" exercise.csv: Exercise data are event-based; each row is an exercise session, which should be assigned to start_day_in_study and 
aggregated to daily exercise features. I should keep the main session per day (longest duration) and also compute daily totals (e.g. total exercise minutes). """

""" subject-info.csv: Subject info is static and not indexed by day_in_study; I should merge it with the daily master using 'id' to add 
static features to each day. """
