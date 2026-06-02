"""Variable-type classification and skewness diagnostics on daily_master_v2.csv
to guide log-transform decisions."""

import pandas as pd # type: ignore
import numpy as np # type: ignore
from scipy.stats import skew # type: ignore 

daily_df = pd.read_csv("data/processed/daily_master_v2.csv")

def variable_diagnostics(df):
 results = []

 # 1 Select numeric columns
 numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

 # 2 Base exclusions
 exclude = {"id", "study_interval", "day_in_study"}

 # 3 Automatically exclude metadata / count variables
 exclude.update([c for c in numeric_cols if c.endswith("_n")])
 exclude.update([c for c in numeric_cols if c.endswith("_n_points")])
 exclude.update([c for c in numeric_cols if c.endswith("_error")])
 exclude.update([c for c in numeric_cols if "count" in c.lower()])

 # 4 Keep only modeling variables
 numeric_cols = [c for c in numeric_cols if c not in exclude]

 for col in numeric_cols:
 x = df[col].dropna()

 if len(x) < 10:
 continue

 zero_pct = (x == 0).mean()
 sk = skew(x)

 med = x.median()
 max_med_ratio = (x.max() / med) if med != 0 else np.nan

 results.append({
 "variable": col,
 "zero_pct": round(zero_pct, 3),
 "skewness": round(sk, 3),
 "max/median": round(max_med_ratio, 2),
 })

 return pd.DataFrame(results).sort_values("skewness", ascending=False)


diag = variable_diagnostics(daily_df)

pd.set_option("display.max_rows", None)
print(diag)