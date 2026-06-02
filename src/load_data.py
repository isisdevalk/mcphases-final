"""Canonical CSV loader for mcPHASES tables (McPhasesLoader).

Encapsulates relative paths under data/raw and convenience methods
for loading single tables, batches, or the project's core set."""

# src/load_data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd # type: ignore


@dataclass(frozen=True)
class DataPaths:
    """Holds project data paths. Assumes you run scripts from the project root."""
    project_root: Path = Path(".")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")

    def raw_path(self, filename: str) -> Path:
        return (self.project_root / self.raw_dir / filename).resolve()

    def processed_path(self, filename: str) -> Path:
        return (self.project_root / self.processed_dir / filename).resolve()


class McPhasesLoader:
    """
    Loader for mcPHASES CSV tables.

    Typical usage:
        loader = McPhasesLoader()
        df = loader.load("sleep.csv")
        tables = loader.load_many(["sleep.csv", "steps.csv"])
    """

    def __init__(self, paths: Optional[DataPaths] = None):
        self.paths = paths or DataPaths()

    def list_raw_csvs(self) -> list[str]:
        """List available .csv files in data/raw."""
        raw_dir = (self.paths.project_root / self.paths.raw_dir)
        if not raw_dir.exists():
            raise FileNotFoundError(
                f"Raw data folder not found: {raw_dir.resolve()}\n"
                "Create it and place the mcPHASES CSVs there (data/raw)."
            )
        return sorted([p.name for p in raw_dir.glob("*.csv")])

    def load(
        self,
        filename: str,
        *,
        dtype: Optional[dict] = None,
        parse_dates: Optional[list[str]] = None,
        keep_default_na: bool = True,
    ) -> pd.DataFrame:
        """
        Load a single CSV from data/raw.

        Tips:
        - Use parse_dates=['date'] or similar if the file has a date column.
        - Use dtype={...} to enforce IDs as strings (often helpful).
        """
        path = self.paths.raw_path(filename)
        if not path.exists():
            available = self.list_raw_csvs()
            raise FileNotFoundError(
                f"File not found: {path}\n\nAvailable CSVs in data/raw:\n"
                + "\n".join(f"  - {x}" for x in available)
            )

        df = pd.read_csv(
            path,
            dtype=dtype,
            parse_dates=parse_dates,
            keep_default_na=keep_default_na,
        )

        # Standardize column names lightly (optional but helpful)
        df.columns = [c.strip() for c in df.columns]
        return df

    def load_many(
        self,
        filenames: Iterable[str],
        *,
        dtype: Optional[dict] = None,
        parse_dates: Optional[list[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Load multiple CSVs; returns dict keyed by filename."""
        out: Dict[str, pd.DataFrame] = {}
        for fn in filenames:
            out[fn] = self.load(fn, dtype=dtype, parse_dates=parse_dates)
        return out

    def load_core_tables(self) -> Dict[str, pd.DataFrame]:
        """
        Convenience: load a sensible 'core' set for your project.

        Adjust this list if your dataset uses slightly different filenames.
        """
        core = [
            "data/raw/active_minutes.csv",
            "data/raw/active_zone_minutes.csv",
            "data/raw/altitude.csv",
            "data/raw/calories.csv",
            "data/raw/computed_temperature.csv",
            "data/raw/demographic_vo2_max.csv",
            "data/raw/distance.csv",
            "data/raw/estimated_oxygen_variation.csv",
            "data/raw/exercise.csv",
            "data/raw/glucose.csv",
            "data/raw/heart_rate_variability_details.csv",
            "data/raw/heart_rate.csv",
            "data/raw/height_and_weight.csv",
            "data/raw/hormones_and_selfreport.csv",
            "data/raw/resting_heart_rate.csv",
            "data/raw/respiratory_rate_summary.csv",
            "data/raw/sleep.csv",
            "data/raw/sleep_score.csv",
            "data/raw/steps.csv",
            "data/raw/stress_score.csv",
            "subject-info.csv",
            "data/raw/time_in_heart_rate_zones.csv",
            "wrist_temperature.csv",
        ]
        return self.load_many(core)


if __name__ == "__main__":
    # Quick smoke test (run: python -m src.load_data from project root)
    loader = McPhasesLoader()
    print("Found CSVs in data/raw:")
    for name in loader.list_raw_csvs():
        print(" -", name)

    # Load one table as a demo
    if "hormones_and_selfreport.csv" in loader.list_raw_csvs():
        df = loader.load("hormones_and_selfreport.csv")
        print("\nLoaded hormones_and_selfreport.csv")
        print("Shape:", df.shape)
        print("Columns:", list(df.columns)[:20], "..." if df.shape[1] > 20 else "")
        print(df.head())
