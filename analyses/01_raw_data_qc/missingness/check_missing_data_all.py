"""Per-CSV missingness report across every file in data/raw."""

# analyses/check_missing_data_all.py

import pandas as pd # type: ignore
from pathlib import Path

raw_dir = Path("data/raw")
out_dir = Path("results/missingness_reports_raw")
out_dir.mkdir(exist_ok=True)

print("\nChecking missing data for all files:\n")

all_reports = []

for csv_file in sorted(raw_dir.glob("*.csv")):
    df = pd.read_csv(csv_file)

    missing_count = df.isna().sum()
    missing_pct = df.isna().mean() * 100

    report = (
        pd.DataFrame({
            "variable": missing_count.index,
            "missing_count": missing_count.values,
            "missing_percent": missing_pct.values,
        })
        .query("missing_count > 0")
        .sort_values("missing_percent", ascending=False)
    )

    if report.empty:
        print(f"[OK] {csv_file.name}: no missing data")
        continue

    print(f"\n--- {csv_file.name} ---")
    print(report.head(10))

    # save per-file report
    report.to_csv(out_dir / f"missing_{csv_file.stem}.csv", index=False)

    report["file"] = csv_file.name
    all_reports.append(report)

# combined report across all files
if all_reports:
    combined = pd.concat(all_reports, ignore_index=True)
    combined.to_csv(out_dir / "missing_all_files_combined.csv", index=False)

print("\nMissingness reports saved to /results/")


