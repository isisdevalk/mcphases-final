"""Per-participant, per-interval coverage check against the expected timeline
from hormones_and_selfreport.csv."""

import pandas as pd  # type: ignore
from pathlib import Path

raw_dir = Path("data/raw")
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# Reference: expected timeline per participant per study_interval
ref = pd.read_csv(raw_dir / "hormones_and_selfreport.csv")
expected = (
    ref.groupby(["id", "study_interval"])["day_in_study"]
       .nunique()
       .rename("expected_days")
       .reset_index()
)

rows = []

def pick_day_col(df: pd.DataFrame) -> str | None:
    # Daily-aligned
    if "day_in_study" in df.columns:
        return "day_in_study"
    # Sleep / nightly summaries
    if "sleep_end_day_in_study" in df.columns:
        return "sleep_end_day_in_study"
    # Exercise sessions
    if "start_day_in_study" in df.columns:
        return "start_day_in_study"
    return None

print("\nBuilding ONE combined missingness report across all CSV files...\n")

for csv_file in sorted(raw_dir.glob("*.csv")):
    df = pd.read_csv(csv_file)

    # Must have id to do per-participant reporting
    if "id" not in df.columns:
        continue

    # Compute NaN missingness per participant (within the rows that exist)
    nan_missing = (
        df.groupby("id")
          .apply(lambda x: x.isna().mean().mean())
          .rename("nan_missing_pct")
          .reset_index()
    )
    nan_missing["nan_missing_pct"] = nan_missing["nan_missing_pct"] * 100

    # Coverage missingness (days present vs expected) if possible
    day_col = pick_day_col(df)

    if day_col is not None and "study_interval" in df.columns:
        observed = (
            df.groupby(["id", "study_interval"])[day_col]
              .nunique()
              .rename("observed_days")
              .reset_index()
        )

        cov = expected.merge(observed, on=["id", "study_interval"], how="left")
        cov["observed_days"] = cov["observed_days"].fillna(0).astype(int)
        cov["coverage_pct"] = (cov["observed_days"] / cov["expected_days"]) * 100

        # Merge in NaN missingness (note: NaN missingness is per id only)
        cov = cov.merge(nan_missing, on="id", how="left")

        cov["file"] = csv_file.name
        cov["day_col_used"] = day_col

        rows.append(cov)

    else:
        # Static / non-alignable to daily timeline: still report NaN missingness per participant
        static = expected[["id", "study_interval", "expected_days"]].copy()
        static["observed_days"] = pd.NA
        static["coverage_pct"] = pd.NA
        static = static.merge(nan_missing, on="id", how="left")
        static["file"] = csv_file.name
        static["day_col_used"] = "None (static/event)"

        rows.append(static)

report = pd.concat(rows, ignore_index=True)

# Sort to make it readable
report = report.sort_values(["file", "id", "study_interval"]).reset_index(drop=True)

out_path = results_dir / "missingness_participants_one_report.csv"
report.to_csv(out_path, index=False)

print(f"Saved: {out_path}")
print("\nPreview (first 20 rows):\n")
print(report.head(20))
