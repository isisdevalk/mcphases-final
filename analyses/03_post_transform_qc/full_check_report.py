# src/qc/qc_daily_master_v3.py
"""
qc_daily_master_v3.py

Quality-control checks for the modeling-ready daily_master_v3_transformed.csv.

What it does (fast, no plots by default):
1) Verifies required key columns and 1 row per person-day
2) Detects duplicate _x/_y suffix columns (merge inflation symptom)
3) Checks hormone safety:
 - warns if raw placeholder zeros might have survived (only possible if you still have *_x/*_y)
 - reports how many exact zeros exist in log-hormones (these are valid: log(1)=0)
 - counts NaNs in hormones
4) Checks exposure/load variables:
 - negative values (should not happen after log1p masking)
 - NaNs introduced by negative masking
5) Checks other numeric sanity:
 - % missing per column (top N)
 - infinite values
6) Writes a QC report CSV + prints a concise console summary

Usage:
 python src/qc/qc_daily_master_v3.py

Inputs:
 data/processed/daily_master_v3_transformed.csv

Outputs:
 results/qc/daily_master_v3_qc_summary.txt
 results/qc/daily_master_v3_missingness_top.csv
 results/qc/daily_master_v3_numeric_issues.csv
"""

from __future__ import annotations

from pathlib import Path
import sys
from datetime import datetime

import numpy as np # type: ignore
import pandas as pd # type: ignore


# Paths
IN_PATH = Path("data/processed/daily_master_v3_transformed.csv")
OUT_DIR = Path("results/qc")
OUT_SUMMARY_TXT = OUT_DIR / "daily_master_v3_qc_summary.txt"
OUT_MISSING_TOP = OUT_DIR / "daily_master_v3_missingness_top.csv"
OUT_NUMERIC_ISSUES = OUT_DIR / "daily_master_v3_numeric_issues.csv"

# Config
KEYS = ["id", "study_interval", "day_in_study"]

HORMONES = ["lh", "estrogen", "pdg"] # expected in clean v3 (no suffixes)
EXPOSURE_LOG1P_VARS = [
 "azm_cardio_minutes",
 "azm_total_minutes",
 "azm_fat_burn_minutes",
 "actmin_moderately",
 "actmin_very",
 "actmin_lightly",
 "steps_daily",
 "exercise_duration_minutes_sum",
 "hrzones_in_default_zone_1",
 "hrzones_in_default_zone_2",
 "hrzones_in_default_zone_3",
 "hrzones_below_default_zone_1",
]

TOP_N_MISSING = 25


def ensure_parent(path: Path) -> None:
 path.parent.mkdir(parents=True, exist_ok=True)


def pct(x: float) -> str:
 return f"{100*x:.2f}%"


def main() -> int:
 if not IN_PATH.exists():
 print(f"[ERROR] Input not found: {IN_PATH}", file=sys.stderr)
 return 1

 OUT_DIR.mkdir(parents=True, exist_ok=True)

 df = pd.read_csv(IN_PATH)

 lines: list[str] = []
 lines.append(f"QC report for: {IN_PATH}")
 lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
 lines.append("")

 # 1) Keys + duplicates
 missing_keys = [k for k in KEYS if k not in df.columns]
 if missing_keys:
 lines.append(f"[FAIL] Missing key columns: {missing_keys}")
 ensure_parent(OUT_SUMMARY_TXT)
 OUT_SUMMARY_TXT.write_text("\n".join(lines))
 return 1

 dup_rows = int(df.duplicated(subset=KEYS).sum())
 lines.append("== Shape & uniqueness ==")
 lines.append(f"Rows: {df.shape[0]:,}")
 lines.append(f"Cols: {df.shape[1]:,}")
 lines.append(f"Duplicate person-days (by {KEYS}): {dup_rows:,}")
 if dup_rows > 0:
 lines.append(" -> [WARN] Duplicate person-days detected. This suggests merge inflation.")
 lines.append("")

 # 2) Suffix columns check (_x/_y)
 x_cols = [c for c in df.columns if c.endswith("_x")]
 y_cols = [c for c in df.columns if c.endswith("_y")]

 lines.append("== Merge suffix columns ==")
 lines.append(f"Columns ending with _x: {len(x_cols)}")
 lines.append(f"Columns ending with _y: {len(y_cols)}")
 if x_cols or y_cols:
 lines.append(" -> [WARN] Found _x/_y columns. Your v3 likely came from a merge with overlapping columns.")
 # Show a few
 show = sorted(set(x_cols[:10] + y_cols[:10]))
 lines.append(f" Examples: {show}")
 lines.append("")

 # 3) Numeric sanity: inf / -inf
 numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
 inf_mask = np.isinf(df[numeric_cols]).any(axis=1) if numeric_cols else np.array([], dtype=bool)
 n_inf_rows = int(inf_mask.sum()) if numeric_cols else 0

 # per-column inf counts
 inf_counts = {}
 for c in numeric_cols:
 inf_counts[c] = int(np.isinf(df[c]).sum())

 inf_cols = {c: n for c, n in inf_counts.items() if n > 0}

 lines.append("== Numeric issues ==")
 lines.append(f"Rows with any +/-inf among numeric cols: {n_inf_rows:,}")
 if inf_cols:
 lines.append(" -> [WARN] Columns containing +/-inf:")
 for c, n in sorted(inf_cols.items(), key=lambda x: -x[1])[:15]:
 lines.append(f" {c}: {n:,}")
 lines.append("")

 # 4) Hormones checks (log-scale)
 lines.append("== Hormones (log-scale) checks ==")
 hormone_found = [h for h in HORMONES if h in df.columns]
 if not hormone_found:
 lines.append(" -> [WARN] None of ['lh','estrogen','pdg'] found in v3. "
 "If you have lh_x/lh_y etc, fix suffix columns before modeling.")
 else:
 for h in hormone_found:
 s = pd.to_numeric(df[h], errors="coerce")
 n_na = int(s.isna().sum())
 n_zero = int((s == 0).sum()) # valid: log(1)=0
 n_pos = int((s > 0).sum())
 n_neg = int((s < 0).sum()) # valid if raw in (0,1)
 lines.append(
 f"{h}: NA={n_na:,} | ==0={n_zero:,} (log(1)) | >0={n_pos:,} | <0={n_neg:,}"
 )

 # If there are raw-like hormone columns still present, check for placeholder zeros
 raw_like = [c for c in df.columns if any(c.lower().startswith(h) for h in HORMONES) and c not in HORMONES]
 raw_zero_warn_cols = []
 for c in raw_like:
 if c.endswith("_x") or c.endswith("_y"):
 s = pd.to_numeric(df[c], errors="coerce")
 if int((s == 0).sum()) > 0:
 raw_zero_warn_cols.append(c)

 if raw_zero_warn_cols:
 lines.append(" -> [WARN] Found zeros in suffixed hormone columns (likely raw placeholders).")
 lines.append(f" Columns: {raw_zero_warn_cols[:10]}{'...' if len(raw_zero_warn_cols)>10 else ''}")
 lines.append("")

 # 5) Exposure/load checks
 lines.append("== Exposure/load checks (log1p outputs) ==")
 exp_found = [c for c in EXPOSURE_LOG1P_VARS if c in df.columns]
 if not exp_found:
 lines.append(" -> [WARN] No configured exposure variables found (check naming).")
 else:
 neg_exp = []
 for c in exp_found:
 s = pd.to_numeric(df[c], errors="coerce")
 # log1p(x) should never be < 0 unless original x in (0, 1) (but these are minutes/steps usually ints)
 n_neg = int((s < 0).sum())
 if n_neg > 0:
 neg_exp.append((c, n_neg))
 if neg_exp:
 lines.append(" -> [WARN] Some exposure/log1p variables have negative values. "
 "This may be OK only if original values were fractional < 1.")
 for c, n in sorted(neg_exp, key=lambda x: -x[1])[:15]:
 lines.append(f" {c}: {n:,} negatives")
 else:
 lines.append("All checked exposure/log1p variables are non-negative (expected for minutes/steps).")
 lines.append("")

 # 6) Missingness summary
 miss = df.isna().mean().sort_values(ascending=False)
 miss_top = miss.head(TOP_N_MISSING).reset_index()
 miss_top.columns = ["variable", "missing_frac"]
 miss_top["missing_pct"] = (miss_top["missing_frac"] * 100).round(2)

 miss_top.to_csv(OUT_MISSING_TOP, index=False)

 lines.append("== Missingness (top) ==")
 for _, r in miss_top.iterrows():
 lines.append(f"{r['variable']}: {r['missing_pct']:.2f}%")
 lines.append("")
 lines.append(f"Saved missingness top-{TOP_N_MISSING}: {OUT_MISSING_TOP}")

 # Add this to qc_daily_master_v3.py
# (Place it after you load df and before writing the summary, e.g., after section 6)

 # 6b) Per-participant missingness checks
 lines.append("== Per-participant missingness checks ==")

 # pick key variables to summarize per participant
 PER_PERSON_VARS = [
 # hormones
 "lh", "estrogen", "pdg",
 # example wearables (edit if you want)
 "hr_bpm_mean", "temp_temperature_diff_from_baseline_mean",
 "sleep_score_overall_score",
 "steps_daily",
 "exercise_duration_minutes_sum",
 "glucose_mean",
 "hrv_rmssd_mean",
 ]

 present = [c for c in PER_PERSON_VARS if c in df.columns]
 if not present:
 lines.append(" -> [WARN] None of PER_PERSON_VARS found in dataset. Check variable names.")
 else:
 # fraction missing per person (id x study_interval)
 grp = df.groupby(["id", "study_interval"], dropna=False)

 per_person = grp[present].apply(lambda g: g.isna().mean()).reset_index()
 # per_person columns: id, study_interval, <vars...> where each is missing fraction

 # Save full table
 OUT_PER_PERSON = OUT_DIR / "daily_master_v3_missingness_per_person.csv"
 per_person.to_csv(OUT_PER_PERSON, index=False)
 lines.append(f"Saved per-person missingness table: {OUT_PER_PERSON}")

 # Summarize "who is basically missing everything" per variable
 # thresholds: % of participants with missingness >= 50% / 80% / 95%
 thresholds = [0.50, 0.80, 0.95]
 summary_rows = []
 n_groups = per_person.shape[0]

 for v in present:
 for t in thresholds:
 n_bad = int((per_person[v] >= t).sum())
 summary_rows.append(
 {
 "variable": v,
 "threshold_missing_frac": t,
 "n_id_interval": n_bad,
 "pct_id_interval": round(100 * n_bad / max(n_groups, 1), 2),
 }
 )

 per_person_summary = pd.DataFrame(summary_rows)

 OUT_PER_PERSON_SUM = OUT_DIR / "daily_master_v3_missingness_per_person_summary.csv"
 per_person_summary.to_csv(OUT_PER_PERSON_SUM, index=False)
 lines.append(f"Saved per-person missingness summary: {OUT_PER_PERSON_SUM}")

 # Print a concise table in the report for the worst variables at 80% missing
 worst80 = (
 per_person_summary[per_person_summary["threshold_missing_frac"] == 0.80]
 .sort_values("pct_id_interval", ascending=False)
 .head(10)
 )

 lines.append("Top variables with many participants missing >= 80% of days:")
 for _, r in worst80.iterrows():
 lines.append(f" {r['variable']}: {r['pct_id_interval']:.2f}% of (id,interval) groups")

 lines.append("")
 
 # 7) Save numeric issue table
 issue_rows = []
 for c in numeric_cols:
 s = pd.to_numeric(df[c], errors="coerce")
 issue_rows.append(
 {
 "variable": c,
 "missing_frac": float(s.isna().mean()),
 "n_inf": int(np.isinf(s).sum()),
 "n_neg": int((s < 0).sum()),
 "n_zero": int((s == 0).sum()),
 "min": float(np.nanmin(s.values)) if s.notna().any() else np.nan,
 "p50": float(np.nanmedian(s.values)) if s.notna().any() else np.nan,
 "p95": float(np.nanpercentile(s.dropna().values, 95)) if s.notna().any() else np.nan,
 "max": float(np.nanmax(s.values)) if s.notna().any() else np.nan,
 }
 )

 issues_df = pd.DataFrame(issue_rows).sort_values("missing_frac", ascending=False)
 issues_df.to_csv(OUT_NUMERIC_ISSUES, index=False)
 lines.append(f"Saved numeric issues table: {OUT_NUMERIC_ISSUES}")

 # 8) Write summary
 ensure_parent(OUT_SUMMARY_TXT)
 OUT_SUMMARY_TXT.write_text("\n".join(lines))

 print("\n".join(lines))
 print(f"\n QC summary written to: {OUT_SUMMARY_TXT}\n")

 return 0


if __name__ == "__main__":
 raise SystemExit(main())