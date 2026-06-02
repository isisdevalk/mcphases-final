# mcPHASES Menstrual Cycle Symptom Dynamics

Master's Internship — Artificial Intelligence (Cognitive Neuroscience)
Student: Isis de Valk
Host: Amsterdam UMC — Medical Psychology, OLVG Hospital - Psychiatry, and VU Amsterdam - Artifical Intelligence

## Overview

This project analyzes longitudinal self-report and wearable data from the
[mcPHASES](https://physionet.org/) dataset to study menstrual-cycle-related
symptom dynamics and individual differences. The pipeline ingests raw Fitbit
exports and self-reported hormone/symptom data, builds a daily-resolution
master dataset, and runs quality-control, distribution, and missingness
diagnostics that feed into the modeling stage.

## Project structure

The pipeline is organized in numbered stages — each stage consumes the
previous stage's output. The matching `analyses/` subfolder runs QC on the
intermediate dataset produced by that stage.

```
mc-phases clean/
├── data/
│   ├── raw/                          # mcPHASES CSV exports (not tracked in git)
│   └── processed/                    # Derived daily-resolution tables
├── src/
│   ├── load_data.py                  # Canonical CSV loader (McPhasesLoader)
│   └── pipeline/                     # Sequential dataset-construction pipeline
│       ├── 01_build_daily_v1/        # Raw → daily_master_v1.csv
│       ├── 02_filter_v2.py           # v1 → v2 (drop high-missingness, height/weight)
│       ├── 03_transform_v3.py        # v2 → v3_transformed (log transforms)
│       └── 04_select_final.py        # v3 → daily_master.csv (final var selection)
├── analyses/                         # QC diagnostics, one folder per pipeline stage
│   ├── 01_raw_data_qc/               # Sanity checks on raw CSVs (pre-build)
│   │   ├── basic_checks/
│   │   ├── distributions/
│   │   └── missingness/
│   ├── 02_post_filter_qc/            # Diagnostics on daily_master_v2.csv
│   │   ├── basic_checks/
│   │   ├── distributions/
│   │   └── missingness/
│   ├── 03_post_transform_qc/         # QC on v3_transformed (collinearity, VIF, ICC)
│   └── 04_post_select_qc/            # Participant missingness on final dataset
├── results/                          # Diagnostic outputs (CSVs and PNGs)
├── requirements.txt
├── .gitignore
└── README.md
```

## Reproducibility

Tested with Python 3.11.

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the raw mcPHASES CSVs in `data/raw/` (see the dataset's README for the
expected filenames; the loader will print the list it finds).

Run scripts from the project root so that relative paths resolve correctly:

```bash
# Build the daily master dataset (run stages in order)
python src/pipeline/01_build_daily_v1/final_daily_aggregation.py
python src/pipeline/02_filter_v2.py
python src/pipeline/03_transform_v3.py
python src/pipeline/04_select_final.py

# Run a diagnostic (example: WP centering QC)
python analyses/05_wp_centering_qc/wp_checks.py
```

## Data

This project uses the **mcPHASES** dataset (v1.0.0) — a longitudinal
record of physiological, hormonal, metabolic, and self-reported
menstrual-health data from 42 participants in the Greater Toronto Area,
collected via Fitbit Sense, Dexcom G6 continuous glucose monitors,
at-home Mira hormone tests, and daily surveys.

- **Access:** Restricted Access on PhysioNet. To obtain the data you
  must (a) be a credentialed PhysioNet user, (b) complete the required
  ethics/CITI training, and (c) sign the dataset's Data Use Agreement
  (DUA). See the dataset page for current requirements.
- **Dataset page:** <https://physionet.org/content/mcphases/1.0.0/>
- **Dataset paper:** Symul, L. *et al.* (2026). A longitudinal dataset of
  physiological, hormonal, metabolic, and self-reported menstrual health
  data. *Scientific Data*.
  <https://www.nature.com/articles/s41597-026-06805-3>
- **Version used in this repository:** v1.0.0
- **License & terms of use:** see the dataset page on PhysioNet.

**No participant-level data is redistributed in this repository.** The
`data/` folder is gitignored. To reproduce the pipeline, obtain v1.0.0
from PhysioNet under the DUA and unpack the files into `data/raw/`
preserving the original folder structure.

## Citation

If you use this code, please cite both the accompanying thesis
(forthcoming) and the mcPHASES dataset:

```bibtex
@dataset{symul2026mcphases,
  author    = {Symul, Laura and others},
  title     = {{mcPHASES: A Dataset of Physiological, Hormonal, and
               Self-reported Events and Symptoms for Menstrual Health
               Tracking with Wearables}},
  year      = {2026},
  version   = {1.0.0},
  publisher = {PhysioNet},
  url       = {https://physionet.org/content/mcphases/1.0.0/}
}
```
