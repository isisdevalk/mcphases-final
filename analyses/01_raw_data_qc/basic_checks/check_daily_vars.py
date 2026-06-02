"""Per-variable QC sweep over raw daily-resolution mcPHASES tables
(duplicates, range violations, type coercions)."""

import pandas as pd  # type: ignore
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("results/qc_daily")
OUT.mkdir(parents=True, exist_ok=True)

KEYS = ["id", "study_interval", "day_in_study"]
TS_KEYS = ["id", "study_interval", "day_in_study", "timestamp"]


# Helpers
def load_csv(name: str) -> pd.DataFrame:
    path = RAW / name
    df = pd.read_csv(path)
    return df


def count_exact_dupes(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


def count_timestamp_dupes(df: pd.DataFrame) -> int:
    if "timestamp" not in df.columns:
        return 0
    return int(df.duplicated(subset=[c for c in TS_KEYS if c in df.columns]).sum())


def count_duplicate_days(df: pd.DataFrame) -> int:
    if not all(k in df.columns for k in KEYS):
        return 0
    return int(df.duplicated(subset=KEYS).sum())


def dedup_by_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns and all(k in df.columns for k in KEYS):
        return df.drop_duplicates(subset=[c for c in TS_KEYS if c in df.columns])
    return df.drop_duplicates()


def to_numeric_safe(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def invalid_mask_numeric(df: pd.DataFrame, col: str, min_v=None, max_v=None, invalid_values=None):
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    s = pd.to_numeric(df[col], errors="coerce")
    mask = pd.Series(False, index=df.index)

    if invalid_values is not None:
        mask |= s.isin(invalid_values)

    if min_v is not None:
        mask |= s < min_v
    if max_v is not None:
        mask |= s > max_v

    return mask


def minutes_columns_guess(df: pd.DataFrame) -> list[str]:
    # Catch the common patterns without you needing to list exact column names.
    cols = []
    for c in df.columns:
        lc = c.lower()
        if any(x in lc for x in ["minute", "minutes", "_min", "mins"]) and c not in KEYS and c != "timestamp":
            cols.append(c)
    return cols


def score_columns_guess(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        lc = c.lower()
        if "score" in lc and c not in KEYS and c != "timestamp":
            cols.append(c)
    return cols


def save_examples(df: pd.DataFrame, path: Path, n=200) -> None:
    df.head(n).to_csv(path, index=False)


# File-specific rules
def qc_active_minutes(df: pd.DataFrame):
    df0 = df.copy()
    # Dedup timestamps if present (safe)
    df0 = dedup_by_timestamp(df0)

    mins_cols = minutes_columns_guess(df0)
    to_numeric_safe(df0, mins_cols)

    invalid = pd.Series(False, index=df0.index)
    # minutes must be 0..1440
    for c in mins_cols:
        invalid |= invalid_mask_numeric(df0, c, min_v=0, max_v=1440)

    return df0, invalid, {"minutes_cols_checked": len(mins_cols)}

def qc_hormones_selfreport(df: pd.DataFrame):
    df0 = df.copy()
    # This is your main timeline; we only do structural QC (not physiology cutoffs).
    df0 = df0.drop_duplicates()

    invalid = pd.Series(False, index=df0.index)

    # Keys must exist & not be missing
    for k in KEYS:
        if k in df0.columns:
            invalid |= df0[k].isna()

    # day_in_study should be >= 1 if present
    if "day_in_study" in df0.columns:
        df0["day_in_study"] = pd.to_numeric(df0["day_in_study"], errors="coerce")
        invalid |= df0["day_in_study"] < 1

    return df0, invalid, {"notes": "structural only (keys, missing, day>=1)"}


def qc_respiration(df: pd.DataFrame):
    df0 = df.copy()
    df0 = dedup_by_timestamp(df0)

    invalid = pd.Series(False, index=df0.index)

    # Drop obvious NO_DATA rows (flag as invalid here)
    if "status" in df0.columns:
        invalid |= df0["status"].astype(str).str.upper().eq("NO_DATA")

    # Columns where 0 is known placeholder (same list you used)
    zero_invalid_cols = [
        "full_sleep_breathing_rate",
        "deep_sleep_breathing_rate",
        "light_sleep_breathing_rate",
        "rem_sleep_breathing_rate",
        "full_sleep_standard_deviation",
        "deep_sleep_standard_deviation",
        "light_sleep_standard_deviation",
        "rem_sleep_standard_deviation",
        "full_sleep_signal_to_noise",
        "deep_sleep_signal_to_noise",
        "light_sleep_signal_to_noise",
        "rem_sleep_signal_to_noise",
    ]
    to_numeric_safe(df0, [c for c in zero_invalid_cols if c in df0.columns])
    for c in zero_invalid_cols:
        if c in df0.columns:
            invalid |= (df0[c] == 0)

    # Breathing rates should be positive if present (but don't upper-bound hard)
    for c in ["full_sleep_breathing_rate", "deep_sleep_breathing_rate", "light_sleep_breathing_rate", "rem_sleep_breathing_rate"]:
        if c in df0.columns:
            invalid |= (pd.to_numeric(df0[c], errors="coerce") <= 0)

    return df0, invalid, {"zero_placeholder_cols_checked": sum(c in df0.columns for c in zero_invalid_cols)}


def qc_resting_hr(df: pd.DataFrame):
    df0 = df.copy()
    df0 = dedup_by_timestamp(df0)

    # find likely RHR columns
    hr_cols = [c for c in df0.columns if any(x in c.lower() for x in ["rest", "rhr", "heart_rate", "heartrate"]) and c not in KEYS]
    to_numeric_safe(df0, hr_cols)

    invalid = pd.Series(False, index=df0.index)

    # Conservative daily plausibility for resting HR: 30..120
    # (If you prefer looser: max 200)
    for c in hr_cols:
        invalid |= invalid_mask_numeric(df0, c, min_v=30, max_v=120)

    return df0, invalid, {"hr_cols_checked": len(hr_cols)}


def qc_sleep_score(df: pd.DataFrame):
    df0 = df.copy()
    df0 = dedup_by_timestamp(df0)

    score_cols = score_columns_guess(df0)
    to_numeric_safe(df0, score_cols)

    invalid = pd.Series(False, index=df0.index)
    # scores 0..100
    for c in score_cols:
        invalid |= invalid_mask_numeric(df0, c, min_v=0, max_v=100)

    return df0, invalid, {"score_cols_checked": len(score_cols)}


def qc_time_in_hr_zones(df: pd.DataFrame):
    df0 = df.copy()
    df0 = dedup_by_timestamp(df0)

    invalid = pd.Series(False, index=df0.index)

    # Two possible formats:
    # A) wide format with minutes columns
    # B) long format with heart_zone_id + total_minutes
    if "heart_zone_id" in df0.columns and "total_minutes" in df0.columns:
        to_numeric_safe(df0, ["total_minutes"])
        # total_minutes must be 0..1440
        invalid |= invalid_mask_numeric(df0, "total_minutes", min_v=0, max_v=1440)
    else:
        mins_cols = minutes_columns_guess(df0)
        to_numeric_safe(df0, mins_cols)
        for c in mins_cols:
            invalid |= invalid_mask_numeric(df0, c, min_v=0, max_v=1440)

    return df0, invalid, {"format": "long" if ("heart_zone_id" in df0.columns and "total_minutes" in df0.columns) else "wide"}


QC_FUNCS = {
    "active_minutes.csv": qc_active_minutes,
    "hormones_and_selfreport.csv": qc_hormones_selfreport,
    "respiratory_rate_summary.csv": qc_respiration,
    "resting_heart_rate.csv": qc_resting_hr,
    "sleep_score.csv": qc_sleep_score,
    "time_in_heart_rate_zones.csv": qc_time_in_hr_zones,
}


# Run QC
rows = []

print("\nRunning DAILY QC checks...\n")

for fname, fn in QC_FUNCS.items():
    print(f"[{fname}] loading...")
    df = load_csv(fname)

    exact_dupes = count_exact_dupes(df)
    ts_dupes = count_timestamp_dupes(df)
    day_dupes = count_duplicate_days(df)

    # Save duplicate-day examples (helpful because daily tables should be 1 row/day)
    if day_dupes > 0 and all(k in df.columns for k in KEYS):
        dup_days = df[df.duplicated(subset=KEYS, keep=False)].copy()
        save_examples(dup_days, OUT / f"{fname.replace('.csv','')}_duplicate_day_examples.csv")

    # File-specific invalid rules
    df_checked, invalid_mask, extra = fn(df)

    n_invalid = int(invalid_mask.sum())
    if n_invalid > 0:
        invalid_rows = df_checked.loc[invalid_mask].copy()
        save_examples(invalid_rows, OUT / f"{fname.replace('.csv','')}_invalid_examples.csv")

    rows.append({
        "file": fname,
        "n_rows": len(df),
        "exact_duplicate_rows": exact_dupes,
        "duplicate_timestamp_rows": ts_dupes,
        "duplicate_day_rows": day_dupes,
        "invalid_rows_flagged": n_invalid,
        "extra": str(extra),
    })

    print(f"  rows: {len(df)} | exact dupes: {exact_dupes} | ts dupes: {ts_dupes} | day dupes: {day_dupes} | invalid flagged: {n_invalid}")

summary = pd.DataFrame(rows)
summary.to_csv(OUT / "qc_daily_summary.csv", index=False)

print("\nSaved QC summary to:", OUT / "qc_daily_summary.csv")
print("Saved example files (if any) to:", OUT)
print("\nDone.\n")
