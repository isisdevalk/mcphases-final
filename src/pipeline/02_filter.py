"""Pipeline stage 02: filter v1 daily master.

Reads daily_master_v1.csv, recodes physiologically implausible values as missing,
drops variables with missingness above THRESHOLD_DROP, joins interval-specific
height/weight, and writes daily_master_v2.csv.
"""

from pathlib import Path

import pandas as pd # type: ignore

RAW = Path("data/raw")
PROC = Path("data/processed")

DAILY_IN = PROC / "daily_master_v1.csv"
HEIGHT_WEIGHT_RAW = RAW / "height_and_weight.csv"
OUT = PROC / "daily_master_v2.csv"

THRESHOLD_DROP = 0.70 # drop variables with >70% missingness
KEYS_NEVER_DROP = {"id", "study_interval", "day_in_study", "is_weekend", "phase"}


# Implausible-value rules
# Values matching these rules are recoded to missing, not row-dropped.
IMPLAUSIBLE_RULES = {
 # Fitbit skin temperature deviation from baseline.
 # Extreme deviations such as -8 are treated as artifacts.
 "temp_temperature_diff_from_baseline_mean": lambda s: (s < -3) | (s > 3),

 # Heart-rate related values: broad physiological sanity bounds.
 "hr_bpm_mean": lambda s: (s < 30) | (s > 220),
 "rhr_value": lambda s: (s < 30) | (s > 140),

 # HRV cannot be negative.
 "hrv_rmssd_mean": lambda s: s < 0,

 # Sleep breathing rate cannot be zero or negative.
 "resp_full_sleep_breathing_rate": lambda s: s <= 0,

 # Sleep and stress scores should remain within expected score ranges.
 "sleep_score_overall_score": lambda s: (s < 0) | (s > 100),
 "wear_stress_score": lambda s: (s < 0) | (s > 100),

 # Activity/load variables cannot be negative.
 "steps_daily": lambda s: s < 0,
 "azm_total_minutes": lambda s: s < 0,
 "azm_cardio_minutes": lambda s: s < 0,
 "azm_fat_burn_minutes": lambda s: s < 0,
 "actmin_moderately": lambda s: s < 0,
 "actmin_very": lambda s: s < 0,
 "actmin_lightly": lambda s: s < 0,
 "exercise_duration_minutes_sum": lambda s: s < 0,

 # Glucose: broad plausible daily median range in mmol/L.
 "glucose_median": lambda s: (s < 2.0) | (s > 20.0),
}


def load_height_weight_interval_specific(raw_path: Path) -> pd.DataFrame:
 """
 Create interval-specific height and weight.

 If study_interval == 2022, use height_2022 / weight_2022.
 If study_interval == 2024, use height_2024 / weight_2024.
 """
 df = pd.read_csv(raw_path)

 for c in df.columns:
 if c != "id":
 df[c] = pd.to_numeric(df[c], errors="coerce")

 rows = []

 for _, row in df.iterrows():
 id_ = row["id"]

 if not pd.isna(row.get("height_2022")) or not pd.isna(row.get("weight_2022")):
 rows.append(
 {
 "id": id_,
 "study_interval": 2022,
 "height_cm_hw": row.get("height_2022"),
 "weight_kg_hw": row.get("weight_2022"),
 }
 )

 if not pd.isna(row.get("height_2024")) or not pd.isna(row.get("weight_2024")):
 rows.append(
 {
 "id": id_,
 "study_interval": 2024,
 "height_cm_hw": row.get("height_2024"),
 "weight_kg_hw": row.get("weight_2024"),
 }
 )

 return pd.DataFrame(rows).drop_duplicates(subset=["id", "study_interval"])


def recode_implausible_values(df: pd.DataFrame) -> pd.DataFrame:
 """
 Recode implausible physiological values to missing.

 This keeps the person-day row but removes only the invalid variable value.
 """
 out = df.copy()

 print("\nRecoding implausible physiological values to missing...")

 total_bad = 0

 for col, rule in IMPLAUSIBLE_RULES.items():
 if col not in out.columns:
 continue

 x = pd.to_numeric(out[col], errors="coerce")
 mask = rule(x) & x.notna()
 n_bad = int(mask.sum())

 if n_bad > 0:
 print(f" - {col}: recoded {n_bad} implausible values to NaN")
 out.loc[mask, col] = pd.NA
 total_bad += n_bad

 if total_bad == 0:
 print("No implausible values detected based on configured rules.")
 else:
 print(f"Total implausible values recoded: {total_bad}")

 return out


def main() -> None:
 print("\nLoading daily_master_v1...")
 df = pd.read_csv(DAILY_IN)
 print("Input shape:", df.shape)

 # Merge interval-specific height/weight
 print("Loading interval-specific height/weight...")
 hw = load_height_weight_interval_specific(HEIGHT_WEIGHT_RAW)

 # Audit columns, temporarily preserved before overwriting.
 for v in ["height_cm", "weight_kg"]:
 if v in df.columns:
 df[f"{v}_orig"] = df[v]

 if "bmi" in df.columns:
 df["bmi_orig"] = df["bmi"]

 merged = df.merge(hw, on=["id", "study_interval"], how="left")

 # Overwrite from interval-specific height/weight when available.
 merged["height_cm"] = merged["height_cm_hw"].combine_first(merged.get("height_cm"))
 merged["weight_kg"] = merged["weight_kg_hw"].combine_first(merged.get("weight_kg"))

 merged = merged.drop(
 columns=[c for c in ["height_cm_hw", "weight_kg_hw"] if c in merged.columns]
 )

 # Propagate within id × interval.
 print("Propagating height/weight within id × study_interval...")
 merged["height_cm"] = merged.groupby(["id", "study_interval"])["height_cm"].transform(
 lambda s: s.ffill().bfill()
 )
 merged["weight_kg"] = merged.groupby(["id", "study_interval"])["weight_kg"].transform(
 lambda s: s.ffill().bfill()
 )

 # Recompute BMI.
 h_m = merged["height_cm"] / 100.0
 merged["bmi"] = merged["weight_kg"] / (h_m * h_m)
 merged.loc[(merged["height_cm"].isna()) | (merged["weight_kg"].isna()), "bmi"] = pd.NA

 # Drop audit columns
 print("Dropping audit columns...")
 audit_cols = [c for c in ["height_cm_orig", "weight_kg_orig", "bmi_orig"] if c in merged.columns]
 merged = merged.drop(columns=audit_cols)
 print("Dropped:", audit_cols)

 # Recode implausible physiological values
 # This must happen before missingness filtering, because recoding values
 # to NaN can change whether a variable exceeds the missingness threshold.
 merged = recode_implausible_values(merged)

 # Drop >70% missingness variables
 print(f"\nDropping variables with >{THRESHOLD_DROP:.0%} missingness...")
 missingness = merged.isna().mean()

 cols_to_drop = [
 c
 for c in missingness.index
 if missingness[c] > THRESHOLD_DROP and c not in KEYS_NEVER_DROP
 ]

 if cols_to_drop:
 print("Will drop:")
 for c in cols_to_drop:
 print(f" - {c}: {missingness[c]:.2%}")
 merged = merged.drop(columns=cols_to_drop)
 else:
 print("None to drop.")

 print("\nOutput shape:", merged.shape)

 PROC.mkdir(parents=True, exist_ok=True)
 merged.to_csv(OUT, index=False)

 print(f"\nSaved: {OUT}")
 print("Done ")


if __name__ == "__main__":
 main()