# F1Insight ML: Technical Report

**Formula 1 Race Outcome Prediction Using Ensemble Machine Learning**

*Model Version: 2.0.0*
*Report Date: February 2026*

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Data Pipeline](#3-data-pipeline)
4. [Feature Engineering](#4-feature-engineering)
5. [Methodology](#5-methodology)
6. [Results](#6-results)
7. [Discussion](#7-discussion)
8. [Limitations and Future Work](#8-limitations-and-future-work)
9. [Test Suite](#9-test-suite)
10. [Appendix](#10-appendix)

---

## 1. Abstract

F1Insight is a machine learning system for predicting Formula 1 race finishing positions using exclusively pre-race information. The system ingests historical race data from the Ergast API (2014--2025) and supplements it with weather and tyre information from the FastF1 library. A 31-feature set spanning qualifying performance, driver and constructor standings, rolling form metrics, circuit characteristics, and weather conditions is constructed with strict temporal leakage safeguards. Four base regression models -- Ridge Regression, Random Forest, Gradient Boosting, and XGBoost -- are tuned via Optuna Bayesian optimisation with a walk-forward expanding-window cross-validation objective and combined through a two-layer stacking ensemble with a Ridge meta-learner. On the held-out 2024--2025 test set (958 race entries), the ensemble achieves a Mean Absolute Error of 3.0853 positions, a Spearman rank correlation of 0.7056, and a position-bucket classification accuracy of 66.6%, beating the grid-position baseline (MAE 3.1253) by 1.3%. Auxiliary classifiers achieve ROC-AUC scores of 0.94 for podium prediction and 0.88 for top-10 prediction. The system also includes a Monte Carlo pit-stop strategy recommendation engine based on KMeans clustering of historical pit-stop patterns.

---

## 2. Introduction

### 2.1 Problem Statement

Predicting Formula 1 race outcomes is a challenging regression and ranking problem. Given only information available before a race begins, the objective is to predict the finishing position of each driver on the grid. This translates to a 20-driver ranking task for every Grand Prix, where the target variable `finish_position` takes integer values from 1 to 20 (or higher for DNF-classified finishes).

### 2.2 Motivation

Formula 1 is one of the most data-rich sports in the world, yet accurate pre-race prediction remains difficult due to the stochastic nature of race events. Existing public models either (a) use race-time features such as pit-stop counts and lap times, which constitute data leakage and inflate reported metrics, or (b) rely on simple heuristics like grid position. This project develops a principled ML pipeline that uses only pre-race features, employs temporal validation to prevent data leakage across seasons, and quantifies the inherent limits of prediction accuracy in the presence of retirements and safety-car incidents.

### 2.3 F1 Prediction Challenges

Several factors make Formula 1 race prediction fundamentally difficult:

- **Did Not Finish (DNF) events**: Mechanical failures, collisions, and incidents cause approximately 35.6% of race entries in the 2024--2025 test set (341 out of 958 entries) to finish far from their expected position. A driver qualifying P1 who retires is typically classified around P16--P20, introducing substantial and irreducible prediction noise.
- **Safety cars and red flags**: These events compress the field and redistribute positions in ways that cannot be anticipated from pre-race data alone.
- **First-lap incidents**: Multi-car collisions at the start redistribute positions in unpredictable patterns.
- **Weather variability**: Sudden rain or changing track temperatures can invert the competitive order.
- **Regulation changes**: Aerodynamic rule changes between seasons alter the competitive hierarchy, making historical data partially obsolete.
- **Strategic variance**: Teams make real-time pit-stop and tyre decisions that significantly alter finishing positions.

These factors combine to create an estimated 1.5--2.0 positions of irreducible noise in any pre-race prediction model.

---

## 3. Data Pipeline

### 3.1 Data Sources

The system ingests data from two primary sources:

| Source | Data Types | Temporal Coverage |
|--------|-----------|-------------------|
| Ergast API (via Jolpi mirror) | Results, qualifying, races, drivers, constructors, circuits, standings, pit stops | 2014--2025 (12 seasons) |
| FastF1 Library | Weather conditions (track/air temperature, humidity, wind speed, rainfall) | 2018--2025 (8 seasons) |

### 3.2 Collection Pipeline

Data collection proceeds through three sequential stages:

```
1. collect_data.py    -->  12 CSV files from Ergast API (raw_dataset/)
2. collect_fastf1.py  -->  Weather and tyre data from FastF1 (raw_dataset/)
3. clean_data.py      -->  Normalised, deduplicated CSVs (cleaned_dataset/)
```

**Collection** (`app/scripts/collect_data.py`): The `F1DataFetcher` class pulls 12 entity types -- results, qualifying, races, drivers, constructors, circuits, driver standings, constructor standings, pit stops, lap times, seasons, and status -- from the Ergast-compatible REST API. Pagination is handled automatically, and data is written to individual CSV files.

**FastF1 enrichment** (`app/scripts/collect_fastf1.py`): For seasons 2018 onwards, session-level weather data is fetched via the FastF1 Python library, which accesses the official F1 timing data. Weather variables include track temperature, air temperature, humidity, wind speed, and a binary rainfall indicator.

**Cleaning** (`app/scripts/clean_data.py`): Column names are standardised to `snake_case`, string identifiers are lowercased and stripped of whitespace, and an `is_dnf` flag is derived from the status column. Duplicate rows are removed and data types are enforced.

### 3.3 Temporal Scope

The dataset spans 12 complete seasons (2014--2025), yielding 5,105 race entries with valid finishing positions. The temporal split is:

| Split | Seasons | Rows | Purpose |
|-------|---------|------|---------|
| Train | 2014--2021 | 3,267 | Model fitting |
| Validation | 2022--2023 | 880 | Hyperparameter tuning and model selection |
| Test | 2024--2025 | 958 | Final, unbiased evaluation |

No random shuffling is applied. All splits are strictly temporal to prevent future information from leaking into the training set.

---

## 4. Feature Engineering

### 4.1 Pre-Race Feature Constraint

A critical design principle of this system is that **all features must be knowable before the race starts**. This eliminates a class of data leakage common in F1 prediction systems, where race-time variables (e.g., number of pit stops, stint durations, lap times) are used as input features despite being causally downstream of the target variable.

### 4.2 Feature Leakage Analysis (v1 vs. v2)

The initial version (v1) of the system included several race-time features that inflated reported metrics:

| v1 Leaked Feature | Why It Leaks |
|--------------------|-------------|
| `total_stops` | Number of pit stops is determined during the race |
| `mean_stop_duration` | Pit stop durations are measured during the race |
| `num_stints` | Number of tyre stints occurs during the race |
| `avg_stint_length` | Average stint length is a race-time measurement |
| `primary_compound` | Actual tyre choices made during the race |
| `num_compounds_used` | Compound diversity is a race-time outcome |

Version 2 (current) removes all of these features and replaces them with pre-race knowable alternatives:

| v2 Replacement Feature | Description |
|------------------------|-------------|
| `historical_avg_stops_at_circuit` | Mean pit stops at this circuit in all prior races |
| `driver_historical_avg_stops` | Mean pit stops by this driver in last 10 races |
| `circuit_avg_positions_gained` | Historical average overtaking at this circuit |
| `season_round_number` | Normalised round number (0--1) for seasonal form effects |
| `constructor_relative_performance` | Normalised team strength (constructor points / max points) |
| `driver_dnf_rate` | Historical DNF fraction for this driver |
| `constructor_dnf_rate` | Historical DNF fraction for this constructor |

### 4.3 Complete Feature Table

The model uses 31 features: 28 numeric and 3 categorical.

| # | Feature | Type | Category | Description |
|---|---------|------|----------|-------------|
| 1 | `qualifying_position` | Numeric | Core | Driver's qualifying session result |
| 2 | `grid_position` | Numeric | Core | Starting grid position (may differ from qualifying due to penalties) |
| 3 | `driver_prior_points` | Numeric | Standings | Driver championship points entering this round |
| 4 | `driver_prior_wins` | Numeric | Standings | Driver race wins entering this round |
| 5 | `driver_prior_position` | Numeric | Standings | Driver championship standing entering this round |
| 6 | `constructor_prior_points` | Numeric | Standings | Constructor championship points entering this round |
| 7 | `constructor_prior_wins` | Numeric | Standings | Constructor race wins entering this round |
| 8 | `constructor_prior_position` | Numeric | Standings | Constructor championship standing entering this round |
| 9 | `historical_avg_stops_at_circuit` | Numeric | Historical | Mean pit stops at this circuit across all prior races |
| 10 | `driver_historical_avg_stops` | Numeric | Historical | Mean pit stops by this driver in last 10 races |
| 11 | `circuit_lat` | Numeric | Circuit | Circuit geographic latitude |
| 12 | `circuit_lng` | Numeric | Circuit | Circuit geographic longitude |
| 13 | `circuit_avg_positions_gained` | Numeric | Circuit | Average positions gained/lost at this circuit historically |
| 14 | `driver_recent_avg_finish` | Numeric | Rolling | Driver's mean finish position over last 5 races |
| 15 | `driver_circuit_avg_finish` | Numeric | Rolling | Driver's mean finish position at this specific circuit |
| 16 | `driver_avg_positions_gained` | Numeric | Rolling | Driver's mean positions gained over last 5 races |
| 17 | `constructor_recent_avg_finish` | Numeric | Rolling | Constructor's mean finish position over last 5 races |
| 18 | `driver_form_trend` | Numeric | Form | Linear regression slope of driver's last 5 finishes (negative = improving) |
| 19 | `gap_to_teammate_quali` | Numeric | Form | Qualifying position gap to teammate (positive = behind teammate) |
| 20 | `season_round_number` | Numeric | Derived | Normalised round number within season (0--1) |
| 21 | `constructor_relative_performance` | Numeric | Derived | Constructor points / max constructor points in the race |
| 22 | `driver_dnf_rate` | Numeric | Reliability | Historical fraction of races where driver did not finish |
| 23 | `constructor_dnf_rate` | Numeric | Reliability | Historical fraction of races where constructor cars did not finish |
| 24 | `track_temp` | Numeric | Weather | Track surface temperature (Celsius) |
| 25 | `air_temp` | Numeric | Weather | Ambient air temperature (Celsius) |
| 26 | `humidity` | Numeric | Weather | Relative humidity (%) |
| 27 | `is_wet_race` | Numeric | Weather | Binary indicator: 1 if rainfall detected, 0 otherwise |
| 28 | `wind_speed` | Numeric | Weather | Wind speed (m/s) |
| 29 | `driver_id` | Categorical | Identity | Unique driver identifier (label-encoded) |
| 30 | `constructor_id` | Categorical | Identity | Unique constructor identifier (label-encoded) |
| 31 | `circuit_id` | Categorical | Identity | Unique circuit identifier (label-encoded) |

All prior-round standings features use only data from rounds strictly less than the current round, preventing within-season temporal leakage. Rolling averages (features 14--17) are computed using an expanding window over all historical races preceding the current race.

---

## 5. Methodology

### 5.1 Temporal Train/Validation/Test Split

The dataset is split by season boundaries with no random shuffling:

```
Train:      2014 ──── 2015 ──── 2016 ──── 2017 ──── 2018 ──── 2019 ──── 2020 ──── 2021
Validation: 2022 ──── 2023
Test:       2024 ──── 2025  (held out, never seen during tuning)
```

This temporal split ensures that no future season information leaks into model training or hyperparameter selection. The test set is never used for any decision-making during development.

### 5.2 Walk-Forward Expanding Window Cross-Validation

Rather than using standard k-fold cross-validation (which would violate temporal ordering), the system employs a **walk-forward expanding window** scheme with 6 folds:

| Fold | Training Period | Validation Year | Train Size | Val Size |
|------|----------------|-----------------|------------|----------|
| 1 | 2014--2017 | 2018 | 1,647 | 420 |
| 2 | 2014--2018 | 2019 | 2,067 | 420 |
| 3 | 2014--2019 | 2020 | 2,487 | 340 |
| 4 | 2014--2020 | 2021 | 2,827 | 440 |
| 5 | 2014--2021 | 2022 | 3,267 | 440 |
| 6 | 2014--2022 | 2023 | 3,707 | 440 |

The training window expands by one season in each successive fold, and the validation set is always the immediately following season. The mean validation MAE across all 6 folds serves as the objective function for hyperparameter optimisation. The 2024--2025 test seasons are never included in any fold.

### 5.3 Optuna Hyperparameter Tuning

Bayesian hyperparameter optimisation is performed using the Optuna framework (Tree-structured Parzen Estimator). Each of the four base regression models is independently tuned with 30 trials, where each trial evaluates a hyperparameter configuration by running the full walk-forward cross-validation and computing the mean validation MAE.

**Tuned hyperparameters per model:**

| Model | Hyperparameter | Search Space | Best Value |
|-------|---------------|-------------|------------|
| **Ridge** | alpha | [0.01, 100.0] (log) | 5.073 |
| **Random Forest** | n_estimators | [100, 400] | 300 |
| | max_depth | [5, 15] | 9 |
| | min_samples_leaf | [5, 30] | 30 |
| | max_features | [0.3, 0.8] | 0.498 |
| **Gradient Boosting** | n_estimators | [100, 400] | 200 |
| | max_depth | [3, 7] | 4 |
| | learning_rate | [0.01, 0.3] (log) | 0.029 |
| | subsample | [0.6, 1.0] | 0.612 |
| | min_samples_leaf | [5, 50] | 42 |
| **XGBoost** | n_estimators | [100, 500] | 350 |
| | max_depth | [3, 8] | 3 |
| | learning_rate | [0.01, 0.3] (log) | 0.014 |
| | subsample | [0.6, 1.0] | 0.628 |
| | colsample_bytree | [0.6, 1.0] | 0.792 |
| | reg_alpha | [0.001, 10.0] (log) | 0.154 |
| | reg_lambda | [0.001, 10.0] (log) | 0.024 |
| | min_child_weight | [1, 10] | 4 |

The baseline walk-forward CV MAE (with default hyperparameters) was **3.81**. After Optuna tuning, the walk-forward CV MAE improved to approximately **3.40**, a reduction of ~10.8%.

### 5.4 Stacking Ensemble Architecture

The final model is a two-layer stacking ensemble:

```
                        ┌──────────────────────────┐
                        │    Input Features (31)    │
                        └─────────┬────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                    │
     ┌────────▼───────┐ ┌────────▼───────┐  ┌────────▼───────┐
     │  Ridge (tuned)  │ │ Random Forest  │  │    Gradient    │
     │  alpha=5.07     │ │ 300 trees, d=9 │  │   Boosting     │
     │                 │ │ leaf=30        │  │ 200 trees, d=4 │
     └────────┬────────┘ └────────┬───────┘  │ lr=0.03        │
              │                   │          └────────┬───────┘
              │                   │                   │
              │    ┌──────────────┼───────────────────┘
              │    │              │
              │    │    ┌────────▼───────┐
              │    │    │   XGBoost      │
              │    │    │ 350 trees, d=3 │
              │    │    │ lr=0.014       │
              │    │    └────────┬───────┘
              │    │             │
    ──────────▼────▼─────────────▼──────────
    │  Layer 2: Ridge Meta-Learner         │
    │  Trained on OOF predictions          │
    │                                      │
    │  Weights:                            │
    │    Ridge:    0.30  (30%)             │
    │    RF:       0.28  (28%)             │
    │    GB:       0.01  ( 1%)             │
    │    XGBoost:  0.44  (44%)             │
    ────────────────┬──────────────────────
                    │
              ┌─────▼──────┐
              │  Predicted  │
              │  Position   │
              └────────────┘
```

**Layer 1 (Base Models):** Four regression models with Optuna-tuned hyperparameters. During the ensemble training phase, out-of-fold (OOF) predictions are generated via the walk-forward cross-validation procedure. For each fold, each base model is trained on the expanding training window and makes predictions on the held-out validation year. These OOF predictions serve as the training data for the meta-learner.

**Layer 2 (Meta-Learner):** A Ridge regression model is trained to combine the four base model predictions. The meta-learner learns optimal blending weights from the OOF predictions, allowing it to assign higher weight to more accurate base models.

**Final deployment:** After the meta-learner is trained, all four base models are retrained on the full train+validation data (2014--2023) and combined with the meta-learner for test-set evaluation and production deployment.

### 5.5 Preprocessing

1. **Numeric features:** Missing values are imputed with 0. Infinite values are replaced with 0. A `StandardScaler` (zero mean, unit variance) is fitted on the training set.
2. **Categorical features:** `driver_id`, `constructor_id`, and `circuit_id` are encoded via `LabelEncoder`. Unknown categories at inference time are mapped to index -1.
3. **Feature matrix:** Scaled numeric features and encoded categorical features are horizontally concatenated into a single NumPy array.

### 5.6 Classification Models

In addition to regression, binary classifiers are trained for two auxiliary tasks:

- **Podium prediction** (`is_podium`): Whether a driver finishes in positions 1--3.
- **Top-10 prediction** (`is_top_10`): Whether a driver finishes in positions 1--10.

Four classifiers are trained per task: Logistic Regression, Random Forest, Gradient Boosting, and XGBoost. Class imbalance is addressed via `class_weight="balanced"` (sklearn models) or `scale_pos_weight` (XGBoost).

---

## 6. Results

### 6.1 Regression Model Comparison

| Model | Train MAE | Val MAE | Test MAE | Test RMSE | Test Spearman | Bucket Accuracy |
|-------|-----------|---------|----------|-----------|---------------|-----------------|
| Ridge | 3.459 | 3.349 | 3.154 | 4.056 | 0.705 | 65.2% |
| Random Forest | 3.204 | 3.417 | 3.144 | 4.032 | 0.710 | 67.0% |
| Gradient Boosting | 3.111 | 3.423 | 3.196 | 4.076 | 0.701 | 65.4% |
| XGBoost | 3.237 | 3.410 | 3.100 | 4.012 | 0.711 | 67.3% |
| **Ensemble** | **3.238** | **--** | **3.085** | **4.037** | **0.706** | **66.6%** |
| *Grid Baseline* | *--* | *--* | *3.125* | *--* | *0.690* | *--* |

The stacking ensemble achieves the lowest test MAE (3.0853), improving over the grid-position baseline by 0.04 positions (1.28%). XGBoost is the strongest individual model (test MAE 3.100), while Gradient Boosting is the weakest on the test set.

### 6.2 Baseline Comparison

| Metric | Grid Baseline | Ensemble | Improvement |
|--------|--------------|----------|-------------|
| Test MAE | 3.1253 | 3.0853 | 0.0400 (1.28%) |
| Test Spearman | 0.6900 | 0.7056 | 0.0156 (2.26%) |

### 6.3 Walk-Forward Cross-Validation Results

| Fold | Val Year | Train Size | Val Size | Train MAE | Val MAE | Val Spearman | Gap |
|------|----------|-----------|----------|-----------|---------|--------------|-----|
| 1 | 2018 | 1,647 | 420 | 1.995 | 3.749 | 0.579 | 1.754 |
| 2 | 2019 | 2,067 | 420 | 2.140 | 3.778 | 0.566 | 1.638 |
| 3 | 2020 | 2,487 | 340 | 2.249 | 4.325 | 0.360 | 2.076 |
| 4 | 2021 | 2,827 | 440 | 2.401 | 3.534 | 0.633 | 1.133 |
| 5 | 2022 | 3,267 | 440 | 2.431 | 3.880 | 0.510 | 1.449 |
| 6 | 2023 | 3,707 | 440 | 2.486 | 3.597 | 0.600 | 1.111 |
| **Mean** | | | | **2.284** | **3.811** | **0.541** | **1.527** |

Notable observations:

- Fold 3 (2020 validation) exhibits the highest MAE and lowest Spearman, likely due to the anomalous COVID-shortened 2020 season with atypical circuits and compressed scheduling.
- The gap narrows in later folds (4--6), consistent with more training data improving generalisation.
- The overall mean gap of 1.527 is typical for tree-based models on noisy targets.

### 6.4 Feature Importance

Feature importance (averaged across Random Forest, Gradient Boosting, and XGBoost):

| Rank | Feature | RF Importance | GB Importance | XGB Importance |
|------|---------|---------------|---------------|----------------|
| 1 | `qualifying_position` | 38.7% | 45.1% | 23.6% |
| 2 | `grid_position` | 21.9% | 14.6% | 12.5% |
| 3 | `constructor_recent_avg_finish` | 10.9% | 7.7% | 6.1% |
| 4 | `driver_recent_avg_finish` | 7.3% | 5.6% | 5.2% |
| 5 | `constructor_relative_performance` | 3.4% | 1.3% | 10.5% |
| 6 | `constructor_prior_points` | 3.7% | 1.5% | 4.3% |
| 7 | `constructor_prior_wins` | 1.5% | -- | 6.4% |
| 8 | `driver_prior_points` | 1.2% | 1.7% | 2.1% |
| 9 | `circuit_id` | 0.9% | 1.6% | -- |
| 10 | `driver_circuit_avg_finish` | 0.9% | 1.5% | -- |

**Key finding:** Qualifying position alone accounts for 24--45% of the model's predictive power across all tree-based models. Combined with grid position, the top two features represent 37--61% of total importance. This confirms the well-known dominance of Saturday qualifying performance in determining Sunday race outcomes.

### 6.5 Classification Results

**Podium Prediction (is_podium):**

| Model | Test Accuracy | Test F1 | Test ROC-AUC |
|-------|--------------|---------|--------------|
| Logistic Regression | 84.6% | 0.643 | 0.929 |
| **Random Forest** | **86.8%** | **0.670** | **0.943** |
| Gradient Boosting | 89.7% | 0.655 | 0.928 |
| XGBoost | 88.7% | 0.638 | 0.929 |

**Top-10 Prediction (is_top_10):**

| Model | Test Accuracy | Test F1 | Test ROC-AUC |
|-------|--------------|---------|--------------|
| **Logistic Regression** | **80.2%** | **0.795** | **0.878** |
| Random Forest | 80.2% | 0.807 | 0.875 |
| Gradient Boosting | 76.5% | 0.769 | 0.837 |
| XGBoost | 75.9% | 0.770 | 0.822 |

The best podium classifier (Random Forest) achieves an ROC-AUC of 0.943, indicating strong discrimination between podium and non-podium finishes. The top-10 classifier achieves 0.878 ROC-AUC, reflecting the greater difficulty of the binary split at the 10th-position boundary.

### 6.6 Position Bucket Accuracy

Predictions are grouped into three position buckets to assess per-tier accuracy:

| Bucket | Ensemble Accuracy | Notes |
|--------|-------------------|-------|
| P1--3 (Podium) | 18.1% | Hardest to predict; small sample, high-variance outcomes |
| P4--10 (Points) | 61.3% | Moderate accuracy; most competitive region of the grid |
| P11--20 (Backmarkers) | 84.9% | Highest accuracy; backmarker teams are consistently slower |
| **Overall** | **66.6%** | Weighted average across all buckets |

The stark accuracy gradient (18.1% to 84.9%) reflects the inherent predictability spectrum: backmarker finishes are largely determined by car performance, while podium outcomes are heavily influenced by race incidents, strategy, and small performance margins.

---

## 7. Discussion

### 7.1 Why MAE Below 2.5 Is Unrealistic

A common question is whether MAE can be driven substantially below 3.0 with better models or more features. Analysis suggests this is unlikely with pre-race features alone:

- **DNF noise:** 35.6% of test entries (341 out of 958) involve a DNF or classified retirement. A driver qualifying P1 who retires is typically classified at P16--P20, introducing an error of 15--19 positions for that entry. This alone adds approximately 0.5 to the overall MAE.
- **Without DNFs:** When DNF entries are excluded, the grid-position baseline MAE drops to approximately 2.58 and the Spearman correlation rises to 0.79. This establishes a tighter performance ceiling for the non-DNF population.
- **Safety cars and incidents:** Safety cars occur in roughly 40--50% of races, compressing the field and redistributing positions by 1--4 places. First-lap incidents can shuffle 5--10 drivers simultaneously.
- **Estimated irreducible noise floor:** Combining DNF effects, safety car redistribution, and first-lap incidents, the irreducible noise floor is estimated at approximately 1.5--2.0 positions of MAE for any pre-race model. An MAE of 2.0--2.5 would likely require real-time data ingestion (live timing, weather radar, tyre degradation telemetry).

### 7.2 Feature Importance Interpretation

The dominance of `qualifying_position` (39% average importance) confirms the well-known dictum in F1 that "qualifying wins races." The grid is the strongest single predictor because:

1. It directly captures current car performance relative to the field.
2. It reflects driver skill at extracting maximum performance in a single-lap context.
3. Overtaking is difficult at many circuits, so starting position has high inertia through the race.

The `constructor_recent_avg_finish` (11%) and `driver_recent_avg_finish` (7%) features capture momentum and current form, complementing the static qualifying snapshot.

### 7.3 Ensemble vs. Individual Models

The ensemble's improvement over the best individual model (XGBoost, MAE 3.100) is marginal at 0.015 positions. The meta-learner weights reveal why:

- **XGBoost: 0.44 (44%)** -- Receives the largest weight, consistent with its best individual performance.
- **Ridge: 0.30 (30%)** -- The linear model contributes diversity by capturing linear trends that tree models may not optimally represent.
- **Random Forest: 0.28 (28%)** -- Contributes additional tree-based diversity.
- **Gradient Boosting: 0.01 (1%)** -- Effectively eliminated by the meta-learner, as its predictions are redundant with XGBoost.

The near-zero weight for Gradient Boosting and the dominance of XGBoost suggest that the ensemble's value lies primarily in the Ridge-XGBoost combination, which blends linear and nonlinear modelling approaches.

### 7.4 Overfitting Control

The train/validation gap for the ensemble is **-0.15**, indicating slight under-fitting (validation performance better than training). This is a desirable outcome and is attributable to:

1. **Aggressive Optuna regularisation:** High `min_samples_leaf` values (30 for RF, 42 for GB), low learning rates (0.014--0.029), and L1/L2 penalties for XGBoost.
2. **Shallow tree depths:** max_depth of 3--9 across models, preventing overfitting to training-set noise.
3. **Walk-forward CV objective:** Tuning against a temporal validation objective inherently penalises overfitting, as the model must generalise to unseen future seasons.

The gap magnitude of 0.15 is well within the success criterion threshold of 0.5, confirming that overfitting is not a concern.

---

## 8. Limitations and Future Work

### 8.1 Current Limitations

1. **No real-time data:** The model cannot account for formation-lap conditions, pre-race tyre selections, or last-minute setup changes.
2. **DNF prediction is implicit:** DNFs are handled as noisy targets rather than modelled explicitly. A dedicated DNF classifier could reduce prediction error for retirement-prone entries.
3. **Limited weather coverage:** FastF1 weather data is only available from 2018 onwards. Pre-2018 weather features are imputed as zero, reducing their effective training signal.
4. **Regulation change blindness:** Major regulation changes (e.g., 2022 ground-effect rules) create distributional shifts that the model must adapt to using limited post-change data.
5. **Sprint race conflation:** Sprint race results and main race results are not distinguished in the current dataset, potentially introducing noise.

### 8.2 Future Work

1. **Explicit DNF modelling:** Train a binary DNF classifier and use its predictions to adjust expected finishing positions. This could reduce MAE by an estimated 0.2--0.4 positions.
2. **Real-time telemetry integration:** Incorporate live timing data (sector times, tyre degradation curves) for in-race prediction updates.
3. **Neural network architectures:** Experiment with recurrent networks (LSTM/GRU) to model sequential driver performance across races, or graph neural networks to capture the competitive structure of the grid.
4. **Expanded historical data:** Extend the dataset to pre-2014 seasons for additional training signal, though older seasons may have limited relevance under modern regulations.
5. **Circuit embedding:** Learn dense circuit representations from layout data (corner types, straight lengths, elevation changes) rather than relying on label-encoded IDs.
6. **Bayesian prediction intervals:** Replace point predictions with posterior distributions (e.g., using NGBoost or Bayesian neural networks) to quantify prediction uncertainty per driver.

---

## 9. Test Suite

The project includes a comprehensive test suite with **77 tests** (59 ML pipeline tests + 18 API tests) organised across 14 test classes.

### 9.1 Test Classes

| Class | Tests | Coverage Area |
|-------|-------|--------------|
| `TestDataLoading` | 4 | CSV file existence, required columns, no future leakage |
| `TestFeatureEngineering` | ~6 | Dataset build, feature count, temporal ordering |
| `TestModelTraining` | ~5 | Model fitting, prediction shape, non-NaN outputs |
| `TestModelValidation` | ~5 | MAE/RMSE/Spearman thresholds, bucket accuracy |
| `TestInference` | ~4 | Single-row prediction, ensemble inference, output range |
| `TestStrategyRecommendation` | ~4 | Monte Carlo simulation, strategy output format |
| `TestEnhancedStrategy` | ~3 | KMeans clustering, tyre compound modelling |
| `TestDataCleaning` | ~4 | Column normalisation, DNF flag, deduplication |
| `TestFeatureLeakage` | ~5 | Verifies no race-time features in feature list |
| `TestWalkForwardCV` | ~4 | Fold count, temporal ordering, expanding window |
| `TestEnsemble` | ~4 | Meta-learner weights, base model count, OOF predictions |
| `TestProductionPickle` | ~3 | Artifact loading, inference from production model |
| `TestSuccessCriteria` | ~4 | MAE < 3.2, Spearman > 0.65, no overfitting, beats baseline |
| `TestAPIEndpoints` | 18 | Health check, prediction endpoints, data endpoints |

### 9.2 Running Tests

```bash
# All tests
pytest tests/ -v

# ML pipeline tests only
pytest tests/test_ml_pipeline.py -v

# API tests only
pytest tests/test_api.py -v

# Specific test class
pytest tests/test_ml_pipeline.py::TestFeatureLeakage -v
```

---

## 10. Appendix

### A. Success Criteria

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Test MAE | < 3.2 | 3.0853 | PASS |
| Test Spearman | > 0.65 | 0.7056 | PASS |
| No overfitting | train/val gap < 0.5 | -0.1527 | PASS |
| Beats grid baseline | MAE < 3.1253 | 3.0853 | PASS |

All four success criteria are satisfied.

### B. Model Artifact Files

| File | Description | Size |
|------|-------------|------|
| `f1insight_production_model.joblib` | Production pickle (ensemble + preprocessor + classifiers) | ~15 MB |
| `ensemble.joblib` | Ensemble artifact (base models + meta-learner + scaler) | ~12 MB |
| `preprocessor.joblib` | StandardScaler + LabelEncoders + feature lists | ~50 KB |
| `model_regression_ridge.joblib` | Tuned Ridge regression model | ~10 KB |
| `model_regression_random_forest.joblib` | Tuned Random Forest model (300 trees) | ~8 MB |
| `model_regression_gradient_boosting.joblib` | Tuned Gradient Boosting model (200 trees) | ~2 MB |
| `model_regression_xgboost.joblib` | Tuned XGBoost model (350 trees) | ~1 MB |
| `model_classification_podium_*.joblib` | Podium classifiers (4 models) | ~1--8 MB each |
| `model_classification_top10_*.joblib` | Top-10 classifiers (4 models) | ~1--8 MB each |
| `evaluation_report.json` | Full training evaluation metrics | ~15 KB |
| `validation_report.json` | Per-race validation results | ~12 KB |

### C. CLI Commands

```bash
# Full data pipeline
python -m app.scripts.collect_data          # Fetch Ergast API data
python -m app.scripts.collect_fastf1        # Fetch FastF1 weather/tyre data
python -m app.scripts.clean_data            # Clean and normalise
python -m app.ml.train                      # Train all models (default: 50 trials)
python -m app.ml.train --trials 30          # Custom trial count
python -m app.ml.train --quick              # Quick mode (10 trials)
python -m app.ml.validate_model             # Run validation on test set
python -m app.ml.export_production_model    # Export production pickle

# API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Testing
pytest tests/ -v                            # All 77 tests
pytest tests/test_ml_pipeline.py -v         # 59 ML tests
pytest tests/test_api.py -v                 # 18 API tests
```

### D. Software Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| scikit-learn | >= 1.3 | Ridge, RF, GB, preprocessing, metrics |
| xgboost | >= 2.0 | XGBoost regressor and classifier |
| optuna | >= 3.0 | Bayesian hyperparameter optimisation |
| pandas | >= 2.0 | Data manipulation and feature engineering |
| numpy | >= 1.24 | Numerical computation |
| scipy | >= 1.10 | Spearman correlation |
| fastf1 | >= 3.0 | F1 timing and weather data |
| fastapi | >= 0.100 | REST API framework |
| joblib | >= 1.3 | Model serialisation |
| pytest | >= 7.0 | Test framework |

### E. Per-Race Test Set Performance (Selected)

| Season | Round | Circuit | MAE | Spearman |
|--------|-------|---------|-----|----------|
| 2024 | 4 | Suzuka | 1.911 | 0.890 |
| 2024 | 10 | Catalunya | 2.122 | 0.941 |
| 2024 | 13 | Hungaroring | 2.449 | 0.959 |
| 2024 | 15 | Zandvoort | 2.258 | 0.925 |
| 2025 | 3 | Suzuka | 2.134 | 0.931 |
| 2025 | 6 | Miami | 1.957 | 0.875 |
| 2025 | 18 | Marina Bay | 2.379 | 0.871 |
| 2024 | 3 | Albert Park | 4.276 | 0.375 |
| 2024 | 17 | Baku | 4.019 | 0.502 |
| 2025 | 15 | Zandvoort | 5.211 | 0.015 |
| 2025 | 22 | Las Vegas | 4.312 | 0.299 |

Best predictions occur at circuits with processional races and predictable outcomes (Suzuka, Catalunya, Hungaroring). Worst predictions occur at circuits with frequent incidents, safety cars, or unusual conditions (Baku, Zandvoort 2025, Las Vegas 2025).

---

*End of Technical Report*
