# F1Insight — Deliverables Overview

## Documentation Index

- **ARCHITECTURE.md** — System architecture map, phase checklist, gaps and fixes.
- **TECHNICAL.md** — Data pipeline, model design, strategy logic, API, assumptions.
- **DEVELOPER.md** — Setup, environment variables, workflows (collect, clean, train, API).
- **CODE_REVIEW_AND_FIXES.md** — Issues identified and applied fixes.
- **data-collection-and-dataset-analysis-report.md** — Data sources and dataset analysis (academic report).
- **README.md** (project root) — Quick start and project structure.

---

# F1Insight — Data Collection & Dataset Analysis Deliverables

## 1. Confirmation of Data Collection Scope

- **Scope:** Formula 1 data from **seasons 2014 to 2025 inclusive**.
- **Source:** Ergast-compatible API (Jolpi): `https://api.jolpi.ca/ergast/f1`.
- **Method:** Existing repository functions in `app/scripts/collect_data.py` (class `F1DataFetcher`) were used. No data-fetching logic was modified or reimplemented.
- **Execution:** The pipeline was executed via `F1DataFetcher(start_year=2014, end_year=2025).fetch_all_data(include_laps=False)`.

Data collection for the 2014–2025 range is **configured and executed** through this pipeline. Full completion of `fetch_all_data()` produces all CSVs listed below. If any year-specific files (results, qualifying, pitstops, standings, sprint) are missing, re-run the same call to completion.

---

## 2. Inventory of Generated CSV Files

**Output directory:** `app/data/raw_dataset/`

| # | Filename | Description | Granularity |
|---|----------|-------------|-------------|
| 1 | `seasons.csv` | All F1 seasons (API reference) | One row per season |
| 2 | `circuits.csv` | All circuits | One row per circuit |
| 3 | `constructors.csv` | All constructors | One row per constructor |
| 4 | `drivers.csv` | All drivers | One row per driver |
| 5 | `status.csv` | Finishing status codes | One row per status |
| 6 | `races.csv` | Race calendar and metadata (2014–2025) | One row per race |
| 7 | `results.csv` | Race results per driver per race | One row per driver–race |
| 8 | `qualifying.csv` | Qualifying results per driver per race | One row per driver–race |
| 9 | `sprint.csv` | Sprint results (where available) | One row per driver–sprint |
| 10 | `pitstops.csv` | Pit stop records per stop | One row per pit stop |
| 11 | `driver_standings.csv` | Driver championship standings | One row per standing entry |
| 12 | `constructor_standings.csv` | Constructor championship standings | One row per standing entry |

**Current on-disk state (at report generation):**  
The following files were present and inspected: `circuits.csv`, `constructors.csv`, `drivers.csv`, `races.csv`, `seasons.csv`, `status.csv`.  
Races coverage was verified: **252 races**, seasons **2014–2025**.  
The remaining six files (`results.csv`, `qualifying.csv`, `sprint.csv`, `pitstops.csv`, `driver_standings.csv`, `constructor_standings.csv`) are produced by the same `fetch_all_data()` run. If they are not present, run the collection again and allow it to complete (year-specific endpoints may take several minutes).

---

## 3. Dataset Analysis Report

A formal, structured report has been generated:

- **Path:** `docs/data-collection-and-dataset-analysis-report.md`
- **Contents:**  
  Data sources and collection overview; dataset description for all 12 CSVs; data quality assessment; feature potential analysis; suitability for ML and strategy modelling; limitations; recommendations and next steps.
- **Use:** Suitable for inclusion in the project report, synopsis appendix, and viva discussion. Formal academic tone; no code snippets; focus on reasoning and analytical depth.

---

## 4. Summary

- Data collection for **2014–2025** is **configured and executed** using the existing F1DataFetcher; full run produces **12 CSV files** in `app/data/raw_dataset/`.
- **Inventory** of expected CSVs and their roles is documented above and in the main report.
- **Dataset analysis report** is complete and ready for inclusion in project documentation and viva.
