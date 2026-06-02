"""Compare per-variable intraclass correlation (ICC) between v2 and v3 datasets."""

# analyses/variance/icc_v2_v3.py

from pathlib import Path
import numpy as np # type: ignore
import pandas as pd # type: ignore

V2_PATH = Path("data/processed/daily_master_v2.csv")
V3_PATH = Path("data/processed/daily_master_v3_transformed.csv")
OUT_PATH = Path("results/variance/icc_v2_v3.csv")

KEYS = ["id", "study_interval", "day_in_study"]

def compute_icc(df: pd.DataFrame, label: str) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in KEYS]

    rows = []

    for col in numeric_cols:
        data = df[[ "id", col ]].dropna()

        if data["id"].nunique() < 5:
            continue

        # Between-person variance
        person_means = data.groupby("id")[col].mean()
        var_between = np.var(person_means, ddof=1)

        # Within-person variance
        var_within = (
            data.groupby("id")[col]
            .var(ddof=1)
            .mean()
        )

        total_var = var_between + var_within

        if total_var == 0 or np.isnan(total_var):
            icc = np.nan
        else:
            icc = var_between / total_var

        rows.append({
            "variable": col,
            "dataset": label,
            "var_between": var_between,
            "var_within": var_within,
            "ICC": icc
        })

    return pd.DataFrame(rows)


def main():
    df_v2 = pd.read_csv(V2_PATH)
    df_v3 = pd.read_csv(V3_PATH)

    icc_v2 = compute_icc(df_v2, "v2_raw")
    icc_v3 = compute_icc(df_v3, "v3_transformed")

    out = pd.concat([icc_v2, icc_v3], ignore_index=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print("\nICC analysis complete.")
    print(f"Saved to: {OUT_PATH}")

    print("\nTop high-ICC variables (v3):")
    print(
        icc_v3.sort_values("ICC", ascending=False)
        .head(10)[["variable", "ICC"]]
        .to_string(index=False)
    )

if __name__ == "__main__":
    main()