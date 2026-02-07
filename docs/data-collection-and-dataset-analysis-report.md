# F1Insight — Data Collection & Dataset Analysis Report

**Final Year Project – Data Collection & Dataset Analysis Chapter**

**Project:** F1Insight – Race Outcome Prediction & Intelligent Strategy Assistance System

**Scope:** Formula 1 data from seasons 2014–2025 (inclusive), Ergast-compatible API (Jolpi), raw CSV persistence.

---

## 1. Data Sources and Collection Overview

### 1.1 API Description

Data is collected from an **Ergast-compatible** API endpoint (Jolpi: `https://api.jolpi.ca/ergast/f1`). The Ergast Motor Racing API provides historical and current Formula 1 data in a structured JSON format. The API is RESTful, returns paginated results where applicable, and uses a consistent schema across endpoints: a root object `MRData` contains metadata (series, limit, offset, total) and a domain-specific table (e.g. `RaceTable`, `ConstructorTable`) whose keys map to entity lists (e.g. `Races`, `Constructors`). The repository does not modify or reimplement the data-fetching logic; it uses the existing collection module that requests these endpoints, flattens nested structures, and writes one CSV per entity type.

### 1.2 Justification for Seasons 2014–2025

The scope is restricted to **seasons 2014 to 2025 inclusive** for the following reasons:

- **Regulatory consistency:** From 2014 onward, Formula 1 entered the hybrid power-unit era. Collecting from 2014 ensures a consistent technical and sporting rule set (power units, fuel limits, tyre regulations) relevant to current and near-future analytics.
- **Relevance to modern F1:** Race outcome prediction and strategy assistance are most valuable when trained and evaluated on data that reflects current car performance, pit-stop strategies, and reliability patterns. The 2014–2025 window balances history and relevance.
- **Data availability and quality:** The chosen API provides complete coverage for this period (races, results, qualifying, pit stops, standings). Extending backwards would introduce earlier eras with different regulations and potentially different granularity.

No expansion or reduction of this range is intended; it is the fixed scope for F1Insight data collection.

### 1.3 Entity-Wise Data Collection

Collection is performed entity by entity:

- **Master and lookup tables:** Seasons, circuits, constructors, drivers, and status codes are fetched once (all-time or bulk). They are not filtered by year in the collection script; the year scope applies to race-level and result-level data.
- **Year-specific data:** Races, race results, qualifying results, sprint results (where available), pit stops, driver championship standings, and constructor championship standings are requested per season (and, for pit stops, per round). Each row in the resulting CSVs corresponds to one logical entity instance (e.g. one driver result per race, one pit stop record per stop).

Sprint races are included only insofar as the existing repository already calls the sprint endpoint for 2014–2025; no additional scope change was introduced.

### 1.4 Alignment with F1Insight Objectives

The collected entities support the project’s goals as follows:

- **Race outcome prediction:** Race results, qualifying, standings, and status codes provide targets (e.g. finishing position, points) and covariates (grid position, constructor/driver form, DNF status).
- **Intelligent strategy assistance:** Pit stop data (lap, stop number, duration) and race results (position, status) support strategy-related features (e.g. number of stops, stint lengths, reliability).
- **Consistent identifiers:** Shared keys (e.g. `season`, `round`, `driverId`, `constructorId`, `circuitId`, `statusId`) allow joining datasets for feature engineering and modelling without changing the raw schema.

---

## 2. Dataset Description

The following subsections describe each CSV produced by the existing collection pipeline: purpose, key columns, granularity, and relationships. Column names reflect the flattened structure (e.g. `Circuit_circuitId`, `Driver_driverId`) as written by the repository.

### 2.1 seasons.csv

- **Purpose:** Reference list of all F1 seasons available in the API.
- **Key columns:** `season` (year), `url` (Wikipedia link).
- **Granularity:** One row per season.
- **Relationships:** Used to validate or filter season scope; race-level tables are linked by `season`.

**Note:** This file contains all seasons returned by the API (e.g. 1950–2026). The 2014–2025 scope is enforced in the collection of races, results, qualifying, pit stops, and standings, not by filtering this file.

### 2.2 circuits.csv

- **Purpose:** Master list of circuits that have hosted F1 events.
- **Key columns:** `circuitId`, `circuitName`, `Location_locality`, `Location_country`, `Location_lat`, `Location_long`, `url`.
- **Granularity:** One row per circuit.
- **Relationships:** Races reference circuits via `Circuit_circuitId` (or equivalent flattened key). Circuit characteristics (e.g. country, layout) can be joined to race-level data for contextual features.

### 2.3 constructors.csv

- **Purpose:** Master list of constructors (teams).
- **Key columns:** `constructorId`, `name`, `nationality`, `url`.
- **Granularity:** One row per constructor.
- **Relationships:** Results, qualifying, sprint, and standings contain constructor identifiers (e.g. `Constructor_constructorId`). Used for team-level form and performance features.

### 2.4 drivers.csv

- **Purpose:** Master list of drivers who have participated in F1.
- **Key columns:** `driverId`, `givenName`, `familyName`, `dateOfBirth`, `nationality`, `permanentNumber`, `code`, `url`.
- **Granularity:** One row per driver.
- **Relationships:** Results, qualifying, sprint, pit stops, and driver standings reference drivers (e.g. `Driver_driverId`). Essential for driver-level targets and covariates.

### 2.5 status.csv

- **Purpose:** Lookup table of finishing/result status codes (e.g. Finished, Accident, +1 Lap).
- **Key columns:** `statusId`, `status`, `count`.
- **Granularity:** One row per status code.
- **Relationships:** Race results reference `statusId` (or flattened equivalent). Used to identify DNFs, classified finishers, and retirements for outcome and reliability features.

### 2.6 races.csv

- **Purpose:** Race calendar and metadata for each event in the selected season range.
- **Key columns:** `season`, `round`, `raceName`, `date`, `time`, `Circuit_circuitId`, `Circuit_circuitName`, `Circuit_Location_*`, and session dates/times (e.g. `FirstPractice_date`, `Qualifying_date`, `Sprint_date` where present).
- **Granularity:** One row per race (one per season–round combination).
- **Relationships:** The primary link for all race-level data. Results, qualifying, sprint, pit stops, and standings can be joined on `season` and `round`. Circuit details can be joined via `Circuit_circuitId`.

**Observed coverage (2014–2025):** 252 races (e.g. 19–24 per season depending on calendar). Season range and race counts align with the stated scope.

### 2.7 results.csv

- **Purpose:** Race result for each driver entry in each race (position, points, status, grid, laps, times).
- **Key columns (typical):** Race keys: `season`, `round`, `raceName`, `date`, `Circuit_*`. Result keys: position, points, grid, laps, status (or `Status_statusId`), and nested Driver/Constructor identifiers and numbers. Time and FastestLap information if provided by the API.
- **Granularity:** One row per driver–race combination (one logical entity per driver per race).
- **Relationships:** Join to `races` on `season` and `round`; to `drivers` and `constructors` on driver and constructor IDs; to `status` on status ID. Primary table for race outcome targets (position, points) and DNF/status flags.

### 2.8 qualifying.csv

- **Purpose:** Qualifying (and where applicable sprint qualifying) result per driver per race (grid-determining session).
- **Key columns (typical):** Race keys plus qualifying position, Q1/Q2/Q3 times (or equivalent), Driver/Constructor identifiers.
- **Granularity:** One row per driver–race combination for the qualifying session.
- **Relationships:** Join to `races`, `drivers`, `constructors`. Used for grid position and qualifying pace as inputs to race outcome and strategy models.

### 2.9 sprint.csv

- **Purpose:** Sprint race results where the format includes a sprint (e.g. 2021 onward in selected rounds).
- **Key columns (typical):** Same structure as race results but for the sprint event (position, points, laps, status, Driver/Constructor).
- **Granularity:** One row per driver–sprint combination.
- **Relationships:** Join to `races` (same `season`, `round`), `drivers`, `constructors`. Optional input for weekends with sprint format; may be empty or sparse for 2014–2020.

### 2.10 pitstops.csv

- **Purpose:** Individual pit stop records (lap, stop number, duration) per driver per race.
- **Key columns (typical):** Race keys plus driver identifier, `lap`, `stop`, `duration` (and possibly `time`).
- **Granularity:** One row per pit stop event.
- **Relationships:** Join to `races` on `season` and `round`, to `drivers`. Used for strategy features: number of stops, stint lengths, stop duration. Many-to-one with race and driver.

### 2.11 driver_standings.csv

- **Purpose:** Championship standings entries per driver per round (or per season-end depending on API).
- **Key columns (typical):** `season`, round or race reference, `position`, `points`, `wins`, Driver identifier.
- **Granularity:** One row per driver–standings snapshot (e.g. per round or per standings list).
- **Relationships:** Join to `drivers` and to `races` or season/round for temporal alignment. Used for form and championship position as features.

### 2.12 constructor_standings.csv

- **Purpose:** Constructor championship standings per round or season.
- **Key columns (typical):** `season`, round or race reference, `position`, `points`, `wins`, Constructor identifier.
- **Granularity:** One row per constructor–standings snapshot.
- **Relationships:** Join to `constructors` and to race/season. Used for team form and competitiveness.

---

## 3. Data Quality Assessment

### 3.1 Missing Data Patterns

- **Optional sessions:** Qualifying and practice session dates/times in `races.csv` can be empty for some rounds (e.g. when format differs or data is absent). Sprint-related columns are empty for seasons or events without sprint races.
- **Result-level gaps:** Not every driver who starts a race may have a complete set of times or fastest-lap data; status codes are the reliable indicator for DNF vs classified finish. Qualifying Q2/Q3 can be missing for drivers eliminated in earlier segments.
- **Pit stops:** Only races where the API provides pit stop data will have rows; some historical rounds may have no or partial pit stop data. Drivers who do not pit (e.g. some shortened races) will have no pit stop rows.
- **Standings:** Snapshot timing (round-by-round vs end-of-season only) depends on the API; missing rounds would appear as gaps in standing-based features.

No cleaning or imputation was applied in this phase; the assessment is analytical only.

### 3.2 DNFs and Race Status Handling

- **Status codes:** The `status` table provides a consistent set of finish reasons (Finished, +1 Lap, Accident, Engine, etc.). Race results reference a status ID, enabling binary or categorical flags (e.g. DNF, classified, retired).
- **Reliability and bias:** High DNF counts for a constructor or driver in a period can bias outcome models if not treated (e.g. separate modelling of “finish vs not finish” and “position given finish”). Status is essential for strategy and outcome modelling.

### 3.3 Limitations of Publicly Available F1 Data

- **No telemetry:** No car telemetry (throttle, brake, speed, ERS) is present; strategy and performance features are limited to aggregated results, times, and pit stops.
- **Weather:** Only coarse or indirect weather information is available if present in the API (e.g. session descriptors); no systematic per-session or per-lap weather.
- **Tyre and compound:** If the API provides compound or stint information in results or pit stops, it can be used; otherwise tyre strategy is inferred only from pit stop counts and lap numbers.
- **Incomplete qualifying or pit stop records:** Some rounds may have partial qualifying data (e.g. only Q1); pit stop coverage may vary by year and circuit.

### 3.4 Consistency of Identifiers

- **Cross-file links:** `season`, `round`, `driverId`, `constructorId`, `circuitId`, `statusId` are designed to be consistent across CSVs. Flattening may produce keys such as `Circuit_circuitId` or `Driver_driverId`; joins require aligning these with the master tables.
- **Stability:** Ergast-style IDs are stable over time, so merges across seasons and datasets are feasible without schema change.

---

## 4. Feature Potential Analysis

### 4.1 Strong Predictors for Race Outcome Prediction

- **Grid position (qualifying):** Strong historical predictor of final position; can be used as a direct feature or via constructor/driver averages.
- **Championship standings (driver and constructor):** Points and position before a race reflect recent form and car performance; useful as rolling or lagged features.
- **Circuit–driver and circuit–constructor history:** Past results at the same circuit (e.g. average finishing position, DNF rate) can be derived from `results` and `races`.
- **Constructor and driver identifiers:** Categorical or embedding inputs for team and driver strength.
- **Status/DNF history:** Prior DNF or retirement rate per driver/constructor as a reliability feature.

### 4.2 Features for Podium Probability Estimation

- **Top-three frequency:** Derived from results (position ≤ 3) by driver/constructor over a rolling window.
- **Qualifying position (e.g. top three):** Indicator or count of recent front-row or top-three starts.
- **Points and wins:** From standings or aggregated results, as form indicators for podium likelihood.

### 4.3 Features Relevant to Strategy Assistance

- **Pit stop count and lap of stops:** From `pitstops.csv`; number of stops and stint lengths per driver per race.
- **Pit stop duration:** Per-stop duration and, if aggregated, average or total pit time per race.
- **Race length and circuit:** From `races`; race distance or lap count supports strategy templates.
- **Reliability (status):** DNF and retirement rates by constructor/driver to inform risk in strategy scenarios.

### 4.4 Features to Exclude or Transform Later

- **Raw URLs and free text:** Wikipedia URLs and long text fields are not used as model inputs without transformation.
- **Identifiers as raw numbers:** `driverId`, `constructorId`, etc. should be encoded (e.g. one-hot, embedding) or used in grouped aggregates rather than as unbounded numeric features.
- **Highly sparse or API-dependent fields:** Optional session times or rarely populated columns should be excluded or handled with missingness indicators.
- **Leakage:** Post-race or end-of-season standings must not be used for in-race prediction; only standings or form known before the race should be used (lagged/prior round).

Conceptually, raw columns map to ML features as follows: identifiers → categorical/embedding or grouping keys; position/grid/points → numeric or ordinal targets/covariates; status → binary/categorical DNF or outcome type; pit stop columns → strategy and stint features; dates → temporal and recency features.

---

## 5. Suitability for Machine Learning and Strategy Modelling

### 5.1 Readiness for Supervised Learning

- **Labelled outcomes:** Race results provide clear labels: finishing position, points, and status (finished/DNF). Qualifying provides grid position. Standings provide cumulative performance. The data is suitable for supervised learning once joined and aggregated to the desired granularity (e.g. one row per driver–race).
- **Preprocessing required (conceptual):** Join results with qualifying, standings, circuits, and optionally pit stops; handle missing qualifying or pit stop data; encode categoricals; create lagged/rolling features (e.g. prior round standings, average position at circuit); possibly separate models for “finish vs DNF” and “position given finish”.

### 5.2 Suitability by Task

- **Classification (e.g. Top 3, Top 10):** Well supported. Target can be derived from `position` (e.g. podium = position ≤ 3). Covariates from qualifying, standings, circuit history, and constructor/driver.
- **Regression (finishing position, points):** Well supported. Same covariates; target is continuous or ordinal (position) or numeric (points). Care with DNFs: either exclude, treat as censored, or model separately.
- **Strategy simulation inputs:** Pit stop counts, lap of stops, and duration support simple strategy rules (e.g. one-stop vs two-stop). Lack of telemetry and detailed tyre compound limits fidelity; strategy simulation would rely on historical patterns and high-level parameters.

### 5.3 Required Preprocessing (Conceptual Only)

- **Merging:** Join races → results, results → qualifying (same driver/race), results → pit stops (same driver/race), and optionally results/standings to driver and constructor master tables. Sprint can be merged for sprint weekends.
- **Missingness:** Define policy for missing qualifying (e.g. back-of-grid indicator), missing pit stops (e.g. zero stops or missing indicator), and optional session data.
- **Temporal:** Ensure no future leakage; use only prior rounds or prior seasons for form and standings.
- **Scaling and encoding:** Numeric features scaled as needed; categoricals (driver, constructor, circuit, status) encoded for the chosen model type.

---

## 6. Limitations of the Current Dataset

- **No telemetry:** Lap-by-lap car data (speed, throttle, brake, ERS) is absent; strategy and performance analysis are limited to outcomes and aggregated times.
- **No real-time race evolution:** Only final results and per-stop pit data; no lap-by-lap position or gap evolution, which would improve strategy and safety-car impact modelling.
- **Limited weather granularity:** No systematic, per-session or per-lap weather; weather-dependent strategy or outcome effects are hard to capture.
- **Dependency on historical patterns:** Predictions rely on past results, qualifying, and standings; structural breaks (regulation changes, new circuits) may require careful validation or segment-specific modelling.
- **API and coverage:** Completeness of pit stops, qualifying segments, and sprint data depends on the Ergast/Jolpi API; gaps or inconsistencies in the source propagate to the raw CSVs.

---

## 7. Recommendations and Next Steps

### 7.1 Preprocessing

- **Next steps:** Implement joins (races, results, qualifying, standings, pit stops) into a single analysis-ready table or star schema; define missingness rules and create derived columns (DNF flag, podium flag, stint length, stop count). Validate identifier consistency across files.
- **No schema change at collection:** Keep raw CSVs as-is; all preprocessing in a separate step or layer.

### 7.2 Merging and Modelling

- **Conceptual merge strategy:** Use `season` and `round` as the race key; join results to qualifying on race + driver; join pit stops to results on race + driver; join standings (prior round) to results on season + driver/constructor. Sprint data merged only for events that have a sprint.
- **ML pipeline:** Use the merged dataset for train/validation/test splits (e.g. by season or by time); implement classification (podium/Top 10) and regression (position/points) baselines; consider separate DNF vs position-given-finish models.

### 7.3 Additional Data Sources

- **FastF1:** Lap times, telemetry, and tyre information would significantly improve strategy and pace modelling; integration would require separate ingestion and alignment with existing `season`/`round`/driver/constructor identifiers.
- **Kaggle or other curated sets:** Weather or tyre compound datasets, if aligned to race keys, could enrich the current data without replacing it.

### 7.4 Integration with F1Insight Components

- **ML training:** Use the preprocessed dataset as the primary input for race outcome and podium models; expose key features and predictions via backend APIs.
- **Strategy simulation:** Use pit stop and result aggregates to parameterise simple strategy models (e.g. stop count, average stint length by circuit or compound if available).
- **Backend APIs:** Serve aggregated statistics, historical results, and model inputs/outputs from the same identifiers (season, round, driver, constructor).
- **Frontend dashboards:** Present race and standings data, predictions, and strategy summaries using the same entity and race keys for consistency.

---

## Summary

Data collection for F1Insight uses the existing repository pipeline against the Ergast-compatible Jolpi API, restricted to seasons **2014–2025**. The pipeline produces master tables (seasons, circuits, constructors, drivers, status) and year-scoped race-level data (races, results, qualifying, sprint, pit stops, driver and constructor standings) as raw CSVs. Races coverage for 2014–2025 is confirmed (252 races). The datasets are suitable for race outcome prediction and high-level strategy assistance, with clear relationships and identifier consistency. Data quality is sufficient for supervised learning after conceptual preprocessing; limitations (no telemetry, limited weather, historical dependency) are documented. Next steps are preprocessing and merging, feature engineering, model training, and optional enrichment from FastF1 or other sources, aligned with the F1Insight objectives.

---

*This report is intended for inclusion in the project report, synopsis appendix, and viva discussion. No code or implementation details are included; emphasis is on design choices, reasoning, and analytical depth.*
