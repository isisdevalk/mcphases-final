"""Test whether missingness in daily variables depends on menstrual-cycle phase
(chi-squared contingency tests)."""

import pandas as pd # type: ignore
import numpy as np # type: ignore
from pathlib import Path
from scipy.stats import chi2_contingency # type: ignore

# Paths
DATA_PATH = Path("data/processed/daily_master_v2.csv")
OUT_DIR = Path("results/missingness")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("\nLoading dataset...")
df = pd.read_csv(DATA_PATH)

PHASE_COL = "phase"
KEYS = ["id", "study_interval", "day_in_study", "is_weekend", PHASE_COL]

# drop rows without phase (can't test phase dependence there)
df = df.dropna(subset=[PHASE_COL]).copy()

# variables to test
value_cols = [c for c in df.columns if c not in KEYS]

print(f"Testing {len(value_cols)} variables for phase-dependent missingness...\n")

# 1) Missingness proportions by phase (descriptive)
missingness_by_phase = (
 df[value_cols].isna()
 .groupby(df[PHASE_COL])
 .mean()
 .reset_index()
)
missingness_by_phase.to_csv(OUT_DIR / "missingness_by_phase.csv", index=False)

# 2) Chi-square tests per variable
results = []

for col in value_cols:
 # missing indicator
 is_missing = df[col].isna().astype(int)

 # contingency table: phase x missing(0/1)
 contingency = pd.crosstab(df[PHASE_COL], is_missing)

 # if everyone is missing or everyone observed, skip
 if contingency.shape[1] < 2:
 continue

 chi2, p, dof, _ = chi2_contingency(contingency)

 results.append({
 "variable": col,
 "chi2_stat": chi2,
 "p_value": p,
 "df": dof,
 "overall_missing_rate": float(is_missing.mean())
 })

tests = pd.DataFrame(results).sort_values("p_value").reset_index(drop=True)

# 3) Multiple testing correction (Benjamini–Hochberg FDR)
def benjamini_hochberg(pvals: pd.Series) -> pd.Series:
 m = len(pvals)
 ranks = pvals.rank(method="first").astype(int)
 qvals = pvals * m / ranks
 # enforce monotonicity
 qvals = qvals.sort_values(ascending=False).cummin().sort_index()
 return qvals

if len(tests) > 0:
 tests["q_fdr_bh"] = benjamini_hochberg(tests["p_value"])
 tests["phase_dependent_fdr_0_05"] = tests["q_fdr_bh"] < 0.05

tests.to_csv(OUT_DIR / "phase_dependent_missingness_tests.csv", index=False)

print("Top 10 variables by smallest p-value:")
print(tests.head(10))

if len(tests) > 0:
 n_sig = int(tests["phase_dependent_fdr_0_05"].sum())
 print(f"\nVariables significant after FDR (q<0.05): {n_sig} / {len(tests)}")

print("\nSaved:")
print(f" - {OUT_DIR / 'missingness_by_phase.csv'}")
print(f" - {OUT_DIR / 'phase_dependent_missingness_tests.csv'}")
print("\nDone ")

phase_missing = (
 df[value_cols].isna()
 .groupby(df["phase"])
 .mean()
)

print(phase_missing[[
 "azm_total_minutes",
 "rhr_value",
 "actmin_moderately", 
 "actmin_very", 
 "actmin_lightly",
 "hr_bpm_n",
 "glucose_n"
]])