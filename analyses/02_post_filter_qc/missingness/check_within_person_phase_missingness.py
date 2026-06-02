"""Within-person mixed-effects models testing whether missingness in daily variables
depends on menstrual-cycle phase."""

import pandas as pd # type: ignore
import numpy as np # type: ignore
from pathlib import Path
import statsmodels.api as sm # type: ignore
import statsmodels.formula.api as smf # type: ignore

# Load data
DATA_PATH = Path("data/processed/daily_master_v2.csv")
OUT_DIR = Path("results/missingness")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("\nLoading dataset...")
df = pd.read_csv(DATA_PATH)

PHASE_COL = "phase"

# drop rows without phase
df = df.dropna(subset=[PHASE_COL]).copy()

KEYS = ["id", "study_interval", "day_in_study", "is_weekend", PHASE_COL]
value_cols = [c for c in df.columns if c not in KEYS]

print(f"Testing {len(value_cols)} variables for within-person phase-dependent missingness...\n")

results = []

# Loop through variables
for col in value_cols:

 # skip variables that are entirely missing or entirely observed
 if df[col].isna().mean() in [0, 1]:
 continue

 temp = df[["id", PHASE_COL, col]].copy()
 temp["missing"] = temp[col].isna().astype(int)

 # need variation
 if temp["missing"].nunique() < 2:
 continue

 try:
 model = smf.gee(
 "missing ~ C(phase)",
 groups="id",
 data=temp,
 family=sm.families.Binomial()
 ).fit()

 # global test: are any phase coefficients significant?
 pvals = model.pvalues.drop("Intercept", errors="ignore")

 if len(pvals) == 0:
 continue

 min_p = pvals.min()

 results.append({
 "variable": col,
 "min_phase_pvalue": float(min_p),
 "overall_missing_rate": float(temp["missing"].mean())
 })

 except Exception as e:
 # some variables may fail due to convergence; skip safely
 continue


results_df = pd.DataFrame(results).sort_values("min_phase_pvalue").reset_index(drop=True)

# FDR correction (Benjamini–Hochberg)
def benjamini_hochberg(pvals):
 m = len(pvals)
 ranks = pvals.rank(method="first")
 qvals = pvals * m / ranks
 qvals = qvals.sort_values(ascending=False).cummin().sort_index()
 return qvals

if len(results_df) > 0:
 results_df["q_fdr_bh"] = benjamini_hochberg(results_df["min_phase_pvalue"])
 results_df["significant_fdr_0_05"] = results_df["q_fdr_bh"] < 0.05

# save
results_df.to_csv(
 OUT_DIR / "within_person_phase_dependent_missingness.csv",
 index=False
)

# Print significant variables
sig = results_df[results_df["significant_fdr_0_05"] == True]

print("\nSignificant within-person phase-dependent missingness (FDR < 0.05):\n")

if len(sig) == 0:
 print("None found.")
else:
 print(sig[["variable", "min_phase_pvalue", "q_fdr_bh", "overall_missing_rate"]])

print(f"\nTotal significant variables: {len(sig)}")
print("\nDone ")