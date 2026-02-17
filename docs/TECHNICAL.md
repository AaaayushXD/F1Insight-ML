# F1Insight ML Service — Technical Documentation

> **Version:** 1.0.0
> **Runtime:** Python 3.10+ / FastAPI
> **ML Framework:** Scikit-learn + XGBoost (Stacking Ensemble)
> **Port:** 8000

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Technology Stack](#2-technology-stack)
3. [Project Structure](#3-project-structure)
4. [Configuration](#4-configuration)
5. [API Reference](#5-api-reference)
6. [Data Pipeline](#6-data-pipeline)
7. [Feature Engineering](#7-feature-engineering)
8. [Model Architecture](#8-model-architecture)
9. [Training Pipeline](#9-training-pipeline)
10. [Inference Pipeline](#10-inference-pipeline)
11. [Strategy Engine](#11-strategy-engine)
12. [Evaluation & Metrics](#12-evaluation--metrics)
13. [Testing](#13-testing)
14. [Deployment](#14-deployment)

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                     FastAPI Application                       │
│                                                              │
│  ┌─────────┐  ┌──────────────┐  ┌─────────────────────────┐ │
│  │ /predict │  │ /predictions │  │ /strategy               │ │
│  │ (single) │  │ /race (grid) │  │ (Monte Carlo simulation)│ │
│  └────┬─────┘  └──────┬───────┘  └────────────┬────────────┘ │
│       │               │                        │              │
│  ┌────▼───────────────▼────┐  ┌───────────────▼────────────┐ │
│  │   Inference Pipeline     │  │   Strategy Engine          │ │
│  │   (Ensemble → Clamp)     │  │   (Monte Carlo + Tyre Sim)│ │
│  └────────────┬────────────┘  └────────────────────────────┘ │
│               │                                              │
│  ┌────────────▼────────────┐                                 │
│  │  Production Model        │                                │
│  │  f1insight_production_   │                                │
│  │  model.joblib (4.0 MB)  │                                │
│  └─────────────────────────┘                                 │
└──────────────────────────────────────────────────────────────┘
```

The ML service provides two core capabilities:
1. **Race Prediction:** Stacking ensemble predicting finish positions and podium probabilities
2. **Strategy Recommendation:** Monte Carlo pit-stop strategy simulation with weather, safety car, and competitor analysis

---

## 2. Technology Stack

| Category | Package | Purpose |
|----------|---------|---------|
| **API Framework** | FastAPI | REST API with auto-documentation |
| **ML Core** | scikit-learn 1.3+ | Ridge, Random Forest, Gradient Boosting |
| **Boosting** | XGBoost 2.0+ | Gradient boosting + podium classifier |
| **Tuning** | Optuna 3.5+ | Bayesian hyperparameter optimization |
| **Interpretability** | SHAP 0.45+ | Feature importance analysis |
| **Data** | Pandas 2.0+, NumPy 1.24+ | Data manipulation |
| **F1 Data** | FastF1 3.4+ | Telemetry, weather, tyre data |
| **Serialization** | Joblib 1.3+ | Model persistence |
| **Statistics** | SciPy 1.10+ | Statistical functions |
| **Visualization** | Matplotlib, Seaborn | Training plots |
| **Testing** | Pytest 7.0+ | Test framework |

---

## 3. Project Structure

```
app/
├── main.py                              # FastAPI app + all 8 endpoints
├── config/
│   └── config.py                        # Paths, temporal splits, feature flags
├── data/
│   ├── raw_dataset/                     # Ergast API CSVs (12 files)
│   ├── cleaned_dataset/                 # Normalized CSVs
│   ├── processed_dataset/               # ML-ready merged dataset
│   └── fastf1_cache/                    # FastF1 session cache
├── ml/
│   ├── build_dataset.py                 # Feature engineering (31 features)
│   ├── train.py                         # Training pipeline + Optuna tuning
│   ├── inference.py                     # Production prediction logic
│   ├── strategy.py                      # Monte Carlo strategy engine (765 lines)
│   ├── export_production_model.py       # Bundle artifacts to .joblib
│   └── outputs/
│       ├── f1insight_production_model.joblib  # Production model (4.0 MB)
│       └── evaluation_report.json       # Complete metrics report
├── scripts/
│   ├── collect_data.py                  # Ergast API data fetcher
│   ├── clean_data.py                    # Data normalization
│   └── collect_fastf1.py               # Weather + tyre data enrichment
├── docs/                                # Documentation files
└── tests/
    ├── test_api.py                      # API endpoint tests (11)
    └── test_ml_pipeline.py              # ML pipeline tests (59)
```

---

## 4. Configuration

**File:** `app/config/config.py`

```python
# Data Sources
API_BASE_URL = "https://api.jolpi.ca/ergast/f1"

# Data Directories
RAW_DATASET_DIR = "app/data/raw_dataset"
CLEAN_DATASET_DIR = "app/data/cleaned_dataset"
PROCESSED_DATASET_DIR = "app/data/processed_dataset"
FASTF1_CACHE_DIR = "app/data/fastf1_cache"

# Temporal Split (strict — no data leakage)
TRAIN_SEASONS = (2014, 2021)      # 3,267 rows
VAL_SEASONS   = (2022, 2023)      # 880 rows
TEST_SEASONS  = (2024, 2025)      # 958 rows

# FastF1 Range (reliable data)
FASTF1_START_YEAR = 2018
FASTF1_END_YEAR = 2025

# Feature Flags
USE_FASTF1_DATA = True
USE_WEATHER_DATA = True
USE_TYRE_DATA = True

RANDOM_STATE = 42
```

---

## 5. API Reference

**Base URL:** `http://localhost:8000`
**Documentation:** `/docs` (Swagger UI), `/redoc` (ReDoc)

### 5.1 Health & Utility

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root — returns `{ message, docs }` |
| GET | `/health` | Health check — returns `{ status: "healthy" }` |

### 5.2 Data Collection

| Method | Endpoint | Params | Description |
|--------|----------|--------|-------------|
| GET | `/collect` | `start_year`, `end_year`, `include_laps` | Trigger Ergast data collection |

### 5.3 Single Prediction

**GET `/predict`**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `season` | int | Yes | Season year (2014-2030) |
| `round_num` | int | Yes | Race round (1-25) |
| `driver_id` | string | Yes | Ergast driver ID |

**Response:**
```json
{
  "predicted_finish_position": 2.45,
  "podium_probability": 0.87
}
```

### 5.4 Race Prediction

**GET `/api/predictions/race`**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `season` | int | Yes | Season year |
| `round` | int | Yes | Race round |

**Response:**
```json
{
  "season": 2024,
  "round": 1,
  "predictions": [
    {
      "driver_id": "verstappen",
      "constructor_id": "red_bull",
      "predicted_finish_position": 1.23,
      "podium_probability": 0.95
    }
  ]
}
```

### 5.5 Strategy Recommendation

**GET `/strategy`**

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `predicted_position_mean` | float | — | 1-20 | Expected finish position |
| `predicted_position_std` | float | 2.0 | 0.1-10 | Position uncertainty |
| `pit_loss_sec` | float | 22.0 | 15-35 | Pit stop time loss |
| `n_simulations` | int | 2000 | 100-10000 | Monte Carlo iterations |
| `circuit_id` | string | — | — | Circuit identifier |
| `race_laps` | int | 56 | 30-80 | Total race laps |
| `track_temp` | float | — | 10-70 | Track temperature (°C) |
| `air_temp` | float | — | 5-50 | Air temperature (°C) |
| `humidity` | float | — | 0-100 | Humidity (%) |
| `rain_probability` | float | 0.0 | 0-1 | Rain likelihood |
| `is_wet_race` | bool | false | — | Wet race flag |
| `gap_to_car_ahead` | float | 1.0 | 0-60 | Gap in seconds |
| `gap_to_car_behind` | float | 1.0 | 0-60 | Gap in seconds |

**Response includes:** `best_strategy`, `strategy_ranking`, `safety_car_analysis`, `weather_impact`, `competitor_analysis`, `compound_strategies`, `tactical_recommendations`, `race_parameters`

### 5.6 Historical Data

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/seasons` | Available seasons list |
| GET | `/api/races?season=X` | Races for a season |
| GET | `/api/drivers?season=X` | Drivers (optionally filtered) |

---

## 6. Data Pipeline

### 6.1 Collection (`collect_data.py`)

**Source:** Ergast-compatible API with exponential backoff (max 5 retries).

**12 CSV entities collected:**
`seasons`, `circuits`, `drivers`, `constructors`, `races`, `results`, `qualifying`, `sprint`, `pitstops`, `driver_standings`, `constructor_standings`, `status`

### 6.2 Cleaning (`clean_data.py`)

- Column normalization to `snake_case`
- ID standardization (lowercase, trimmed)
- Duration parsing (`mm:ss.fff` → seconds)
- DNF status detection (`status_id != 1`)
- Foreign key validation
- Type consistency enforcement

### 6.3 FastF1 Enrichment (`collect_fastf1.py`)

Adds weather and tyre data for 2018-2025:
- Track temperature, air temperature, humidity
- Rainfall indicator, wind speed
- Tyre compound per stint

### 6.4 Pipeline Commands

```bash
python -m app.scripts.collect_data       # Fetch from Ergast
python -m app.scripts.clean_data         # Normalize
python -m app.scripts.collect_fastf1     # Weather enrichment
python -m app.ml.build_dataset           # Feature engineering
python -m app.ml.train                   # Train models
python -m app.ml.export_production_model # Export for production
```

---

## 7. Feature Engineering

**File:** `app/ml/build_dataset.py`
**Dataset:** One row per `(season, round, driver_id)` — 5,105 total rows

### 31 Features (28 Numeric + 3 Categorical)

| # | Feature | Category | Description |
|---|---------|----------|-------------|
| 1 | `qualifying_position` | Core | Qualifying session result |
| 2 | `grid_position` | Core | Starting grid position |
| 3 | `driver_prior_points` | Driver Standing | Championship points before race |
| 4 | `driver_prior_wins` | Driver Standing | Wins before race |
| 5 | `driver_prior_position` | Driver Standing | Standing position before race |
| 6 | `constructor_prior_points` | Constructor | Team points before race |
| 7 | `constructor_prior_wins` | Constructor | Team wins before race |
| 8 | `constructor_prior_position` | Constructor | Team standing before race |
| 9 | `historical_avg_stops_at_circuit` | Pit Stop | Average stops at this circuit |
| 10 | `driver_historical_avg_stops` | Pit Stop | Driver average stops (last 10) |
| 11 | `circuit_lat` | Circuit | Latitude |
| 12 | `circuit_lng` | Circuit | Longitude |
| 13 | `circuit_avg_positions_gained` | Circuit | Historical overtaking metric |
| 14 | `driver_recent_avg_finish` | Rolling | Avg finish (last 5 races) |
| 15 | `driver_circuit_avg_finish` | Rolling | Avg finish at this circuit |
| 16 | `driver_avg_positions_gained` | Rolling | Avg positions gained (last 5) |
| 17 | `constructor_recent_avg_finish` | Rolling | Team avg finish (last 5) |
| 18 | `driver_form_trend` | Form | Slope of recent 5 finishes |
| 19 | `gap_to_teammate_quali` | H2H | Qualifying gap to teammate |
| 20 | `season_round_number` | Derived | Normalized round (0-1) |
| 21 | `constructor_relative_performance` | Derived | Normalized team strength |
| 22 | `driver_dnf_rate` | Reliability | Historical DNF rate |
| 23 | `constructor_dnf_rate` | Reliability | Team DNF rate |
| 24 | `track_temp` | Weather | Track temperature (°C) |
| 25 | `air_temp` | Weather | Air temperature (°C) |
| 26 | `humidity` | Weather | Humidity (%) |
| 27 | `is_wet_race` | Weather | Binary wet indicator |
| 28 | `wind_speed` | Weather | Wind speed (m/s) |
| 29 | `driver_id` | Categorical | Label-encoded driver |
| 30 | `constructor_id` | Categorical | Label-encoded team |
| 31 | `circuit_id` | Categorical | Label-encoded circuit |

### Target Variables

| Target | Type | Description |
|--------|------|-------------|
| `finish_position` | Regression | Predicted finish (1-20+) |
| `is_podium` | Binary | Position ≤ 3 |
| `is_top_10` | Binary | Position ≤ 10 |
| `is_dnf` | Binary | DNF flag |

### No Future Leakage Guarantee

- Prior standings: Only from `round < current_round` (same season) or prior season end
- Rolling averages: Only from past races
- Circuit history: Only from past events at that circuit
- Qualifying/grid: Current race only (known pre-race)

---

## 8. Model Architecture

### 8.1 Stacking Ensemble (Best Model)

```
Input Features (31)
        │
        ├──▶ Ridge Regression ──────────┐
        ├──▶ Random Forest ─────────────┤
        ├──▶ Gradient Boosting ─────────┤  Out-of-Fold
        └──▶ XGBoost ──────────────────┤  Predictions
                                        │
                                 ┌──────▼──────┐
                                 │ Ridge Meta-  │
                                 │ Learner      │
                                 └──────┬──────┘
                                        │
                                 Clamp [1, 20]
                                        │
                              predicted_finish_position
```

**Meta-Learner Weights:** Ridge(0.303), RF(0.282), GB(0.006), XGB(0.438)

XGBoost receives the highest weight, followed by Ridge and Random Forest. Gradient Boosting is effectively excluded by the meta-learner.

### 8.2 Podium Classifier

Separate XGBoost binary classifier trained on `is_podium` target.

- Input: Same 31 features
- Output: `P(podium)` via `predict_proba`
- ROC-AUC: **0.929** on test set

### 8.3 Tuned Hyperparameters

**Ridge:**
- `alpha`: 5.07

**Random Forest:**
- `n_estimators`: 300, `max_depth`: 9, `min_samples_leaf`: 30, `max_features`: 0.498

**Gradient Boosting:**
- `n_estimators`: 200, `max_depth`: 4, `learning_rate`: 0.0295, `subsample`: 0.612, `min_samples_leaf`: 42

**XGBoost:**
- `n_estimators`: 350, `max_depth`: 3, `learning_rate`: 0.0139, `subsample`: 0.628, `colsample_bytree`: 0.792, `reg_alpha`: 0.154, `reg_lambda`: 0.024, `min_child_weight`: 4

---

## 9. Training Pipeline

**File:** `app/ml/train.py` (851 lines)

### 9.1 Temporal Split

| Split | Seasons | Rows | Purpose |
|-------|---------|------|---------|
| Train | 2014–2021 | 3,267 | Model training |
| Validation | 2022–2023 | 880 | Hyperparameter tuning |
| Test | 2024–2025 | 958 | Final evaluation (never touched during training) |

### 9.2 Walk-Forward Cross-Validation

6-fold expanding window:
- Fold 1: Train 2014-2017 → Validate 2018
- Fold 2: Train 2014-2018 → Validate 2019
- Fold 3: Train 2014-2019 → Validate 2020
- Fold 4: Train 2014-2020 → Validate 2021
- Fold 5: Train 2014-2021 → Validate 2022
- Fold 6: Train 2014-2022 → Validate 2023

### 9.3 Preprocessing

**Numeric:** StandardScaler (fitted on training data only)
**Categorical:** LabelEncoder per column (unknown categories → -1)
**Missing Values:** NaN → 0, Inf → 0

### 9.4 Hyperparameter Optimization

- **Framework:** Optuna (Bayesian TPE sampler)
- **Objective:** Minimize MAE on validation set
- **Trials:** Per-model tuning on walk-forward folds

---

## 10. Inference Pipeline

**File:** `app/ml/inference.py`

```
Input Row (season, round, driver_id)
        │
        ▼
Load Production Model (.joblib)
        │
        ▼
Preprocess: Scale (numeric) + Encode (categorical)
        │
        ▼
4 Base Model Predictions
        │
        ▼
Ridge Meta-Learner → Weighted Combination
        │
        ▼
Clamp to [1, 20]
        │
        ▼
Podium Classifier → P(podium)
        │
        ▼
Output: { predicted_finish_position, podium_probability }
```

**Production Artifact:** `f1insight_production_model.joblib` (4.0 MB)

Contains: scaler, encoders, 4 base models, meta-learner, podium classifier, metadata.

---

## 11. Strategy Engine

**File:** `app/ml/strategy.py` (765 lines)

### 11.1 Tyre Compound Model

| Compound | Pace Advantage | Degradation | Optimal Stint | Max Stint |
|----------|----------------|-------------|---------------|-----------|
| SOFT | +0.8s/lap | 0.08 pos/lap | 15 laps | 22 laps |
| MEDIUM | baseline | 0.04 pos/lap | 25 laps | 35 laps |
| HARD | -0.5s/lap | 0.02 pos/lap | 35 laps | 50 laps |
| INTERMEDIATE | -3.0s/lap | 0.01 pos/lap | 40 laps | 60 laps |
| WET | -6.0s/lap | 0.005 pos/lap | 50 laps | 70 laps |

### 11.2 Safety Car Analysis

- Based on historical incident data per circuit
- SC-causing statuses: Accident, Collision, Spun off, Collision damage
- Formula: `races_with_incidents / total_races × 1.2`
- Default probability: ~35%
- VSC probability: ~25%

### 11.3 Weather Impact

| Condition | Weather Factor | Degradation Multiplier |
|-----------|---------------|----------------------|
| Dry | 1.0 | 1.0 |
| Light rain (prob > 0.3) | 1.3 | 1.0 |
| Heavy rain / Wet | 1.5 | varies |
| High temp (> 50°C) | 1.0 | 1.4 |
| Moderate (40-50°C) | 1.0 | 1.2 |
| Low temp (< 25°C) | 1.0 | 0.8 |
| High humidity (> 80%) | +10% | 1.0 |

### 11.4 Undercut/Overcut Analysis

**Undercut:**
- Fresh tyre advantage: `0.7s/lap × 3 laps = 2.1s` gain
- Viable if: undercut gain > gap ahead AND gap < 3.0s
- Confidence: `min(1.0, gain / gap)`

**Overcut:**
- Viable if: gap behind > 2.5s AND position > 1
- Confidence: `min(1.0, gap / 5.0)`

### 11.5 Strategy Generation

Generates all valid 1-stop and 2-stop strategies from available compounds:

- **1-stop:** All compound pairs, stint lengths based on optimal + degradation factor
- **2-stop:** All compound triplets, stint lengths × 0.8 factor

### 11.6 Strategy Scoring

| Component | Weight | Description |
|-----------|--------|-------------|
| Pace Score | 70% | Faster compounds + coverage + SC benefit |
| Risk Score | 30% | Overpush penalties + degradation exposure |

### 11.7 Monte Carlo Simulation

**Default:** 2,000 iterations per strategy

```
For each iteration:
  1. base_pos = Normal(mean, std)
  2. Add: degradation_noise, traffic_noise
  3. For each strategy:
     - pit_penalty = stops × pit_loss × position_factor
     - If safety car: penalty × 0.3
     - strategy_noise = Normal(0, sqrt(stops × 0.5))
     - final = clip(base + penalty + noise, 1, 20)
  4. Compute: mean, std, percentiles, best/worst case
```

**Output:** Ranked strategies with confidence intervals and recommendations.

---

## 12. Evaluation & Metrics

### 12.1 Regression Performance

| Model | Test MAE | Test Spearman | Train MAE |
|-------|----------|---------------|-----------|
| Ridge | 3.154 | 0.705 | 3.307 |
| Random Forest | 3.144 | 0.710 | 3.238 |
| Gradient Boosting | 3.196 | 0.701 | 3.265 |
| XGBoost | 3.100 | 0.711 | 3.105 |
| **Ensemble** | **3.085** | **0.706** | **3.238** |
| Grid Position Baseline | 3.125 | — | — |

**Ensemble improvement over baseline:** 1.3% (0.040 MAE reduction)

### 12.2 Classification Performance

| Task | Best Model | Accuracy | ROC-AUC | F1-Score |
|------|-----------|----------|---------|----------|
| Podium (≤ 3) | XGBoost | 88.7% | **0.929** | 0.638 |
| Top-10 (≤ 10) | Random Forest | 80.2% | 0.875 | 0.807 |

### 12.3 Position Bucket Accuracy

| Bucket | Accuracy | Interpretation |
|--------|----------|----------------|
| P1-3 (Podium) | 18.1% | High variance at front |
| P4-10 (Points) | 61.3% | Good midfield prediction |
| P11-20 (Outside points) | 84.9% | Very stable backmarkers |
| **Overall** | **66.6%** | |

### 12.4 Overfitting Check

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Train-Test MAE Gap | -0.153 | < 0.5 | Pass |
| Walk-Forward Mean Gap | 1.527 | — | Expected |

Negative gap means test performance is slightly *better* than train — no overfitting.

### 12.5 Feature Importance (Random Forest)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `qualifying_position` | 0.387 |
| 2 | `grid_position` | 0.219 |
| 3 | `constructor_recent_avg_finish` | 0.109 |
| 4 | `driver_recent_avg_finish` | 0.073 |
| 5 | `constructor_prior_points` | 0.037 |

Qualifying and grid position dominate (60%+ combined). Team performance is more predictive than individual driver performance.

### 12.6 Success Criteria (All Passed)

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Test MAE | < 3.2 | 3.085 | Pass |
| Spearman | > 0.65 | 0.706 | Pass |
| No Overfitting | gap < 0.5 | -0.153 | Pass |
| Beat Baseline | < 3.125 | 3.085 | Pass |

---

## 13. Testing

### Test Suite

| File | Tests | Coverage |
|------|-------|----------|
| `test_api.py` | 11 | All endpoints |
| `test_ml_pipeline.py` | 59 | Data, features, training, inference, strategy |
| **Total** | **70** | |

### Test Categories

**API Tests:** Health, root, seasons, races, single prediction, race prediction, strategy (basic, enhanced, parameters, validation, wet conditions, competitor analysis)

**ML Pipeline Tests:**
- Data loading and CSV existence
- Feature engineering (31 features, 4 targets)
- No future data leakage verification
- Temporal split correctness
- Model training and convergence
- Inference pipeline end-to-end
- Strategy recommendation output

### Running Tests
```bash
pytest tests/ -v                    # All tests
pytest tests/test_api.py -v         # API only
pytest tests/test_ml_pipeline.py -v # ML pipeline only
```

---

## 14. Deployment

### Quick Start

```bash
# Environment setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Data pipeline (first time only)
python -m app.scripts.collect_data
python -m app.scripts.clean_data
python -m app.ml.build_dataset
python -m app.ml.train
python -m app.ml.export_production_model

# Start API server
bash run.sh
# or: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Deployment

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Documentation

| URL | Format |
|-----|--------|
| `http://localhost:8000/docs` | Swagger UI |
| `http://localhost:8000/redoc` | ReDoc |
| `http://localhost:8000/openapi.json` | OpenAPI JSON |

### Model Summary

| Attribute | Value |
|-----------|-------|
| Production File | `f1insight_production_model.joblib` |
| File Size | 4.0 MB |
| Architecture | 4-model stacking ensemble + Ridge meta-learner |
| Training Data | 3,267 rows (2014-2021) |
| Test MAE | 3.085 |
| Podium ROC-AUC | 0.929 |
| Features | 31 (28 numeric + 3 categorical) |
| Temporal Integrity | Verified (no future leakage) |
