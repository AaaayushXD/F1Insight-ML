# Technical Documentation: Race Finish Position Prediction Model

## Overview

The F1Insight race prediction system uses machine learning to predict the finishing position of drivers in Formula 1 races. The system employs multiple regression models trained on historical race data with engineered features capturing driver performance, constructor strength, and race conditions.

## Problem Definition

### Objective
Predict the finishing position (1-20) for each driver in an upcoming F1 race.

### Task Type
- **Primary**: Regression (finish position as continuous value)
- **Secondary**: Classification (podium probability, top-10 probability)

### Target Variable
- `finish_position`: Integer from 1 to 20 (or higher for DNF)

## Data Sources

### Primary Data (Ergast API)
| Dataset | Records | Key Fields |
|---------|---------|------------|
| results | ~5,000 | season, round, driver_id, position, grid, points |
| qualifying | ~5,000 | season, round, driver_id, qualifying_position |
| driver_standings | ~2,500 | season, round, driver_id, points, wins, position |
| constructor_standings | ~1,200 | season, round, constructor_id, points, wins |
| pitstops | ~40,000 | season, round, driver_id, lap, stop_number, duration |
| circuits | ~80 | circuit_id, lat, lng, country |

### Enhanced Data (FastF1)
| Dataset | Records | Key Fields |
|---------|---------|------------|
| tyre_stints | ~15,000 | season, round, driver_id, compound, stint_number, laps |
| weather | ~250 | season, round, track_temp, air_temp, humidity, rainfall |

## Feature Engineering

### Feature Categories

#### 1. Position Features (2)
| Feature | Type | Description |
|---------|------|-------------|
| qualifying_position | numeric | Q3/Q2/Q1 result (1-20) |
| grid_position | numeric | Starting grid position |

#### 2. Driver Standing Features (3)
| Feature | Type | Description |
|---------|------|-------------|
| driver_prior_points | numeric | Championship points before race |
| driver_prior_wins | numeric | Wins in season before race |
| driver_prior_position | numeric | Championship position before race |

#### 3. Constructor Standing Features (3)
| Feature | Type | Description |
|---------|------|-------------|
| constructor_prior_points | numeric | Team points before race |
| constructor_prior_wins | numeric | Team wins before race |
| constructor_prior_position | numeric | Team championship position |

#### 4. Pit Stop Features (2)
| Feature | Type | Description |
|---------|------|-------------|
| total_stops | numeric | Number of pit stops in race |
| mean_stop_duration | numeric | Average pit stop time (seconds) |

#### 5. Circuit Features (2)
| Feature | Type | Description |
|---------|------|-------------|
| circuit_lat | numeric | Circuit latitude |
| circuit_lng | numeric | Circuit longitude |

#### 6. Rolling Performance Features (4)
| Feature | Type | Description |
|---------|------|-------------|
| driver_recent_avg_finish | numeric | Mean finish of last 5 races |
| driver_circuit_avg_finish | numeric | Mean finish at this circuit |
| driver_avg_positions_gained | numeric | Mean grid-to-finish delta |
| constructor_recent_avg_finish | numeric | Team's mean finish last 5 races |

#### 7. Form and Head-to-Head Features (2)
| Feature | Type | Description |
|---------|------|-------------|
| driver_form_trend | numeric | Slope of recent finishes (negative = improving) |
| gap_to_teammate_quali | numeric | Qualifying gap to teammate |

#### 8. Weather Features (5)
| Feature | Type | Description |
|---------|------|-------------|
| track_temp | numeric | Track temperature (°C) |
| air_temp | numeric | Air temperature (°C) |
| humidity | numeric | Relative humidity (%) |
| is_wet_race | numeric | Binary wet race indicator |
| wind_speed | numeric | Wind speed (km/h) |

#### 9. Tyre Strategy Features (4)
| Feature | Type | Description |
|---------|------|-------------|
| num_stints | numeric | Number of tyre stints |
| num_compounds_used | numeric | Number of different compounds |
| avg_stint_length | numeric | Average laps per stint |
| total_tyre_laps | numeric | Total laps on all tyres |

#### 10. Categorical Features (4)
| Feature | Type | Description |
|---------|------|-------------|
| driver_id | categorical | Driver identifier (slug) |
| constructor_id | categorical | Team identifier (slug) |
| circuit_id | categorical | Circuit identifier (slug) |
| primary_compound | categorical | Most-used tyre compound |

### Feature Count Summary
- **Numeric Features**: 27
- **Categorical Features**: 4
- **Total Features**: 31

## Data Leakage Prevention

### Temporal Constraints
All features are computed using only data available **before** the race:

1. **Prior Standings**: Uses standings from round < current_round
2. **Rolling Averages**: Only includes races with `_ord < current_ord`
3. **Form Trend**: Computed from historical finishes only

### Implementation
```python
# Prior standings lookup
same_season = standings[
    (standings["season"] == season) &
    (standings["round"] < round_num)
]

# Rolling average calculation
ord_cur = season * 1000 + round_num
past = results[results["_ord"] < ord_cur]
```

## Model Architecture

### Models Trained

| Model | Type | Key Hyperparameters |
|-------|------|---------------------|
| Ridge Regression | Linear | alpha=1.0 |
| Random Forest | Ensemble | n_estimators=100, max_depth=10 |
| Gradient Boosting | Ensemble | n_estimators=100, max_depth=5 |
| XGBoost | Gradient Boosting | n_estimators=100, max_depth=6 |

### Preprocessing Pipeline

```python
# Numeric features
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

# Categorical features
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X_cat_encoded[:, i] = le.fit_transform(X_cat[col])
    encoders[col] = le

# Combined feature matrix
X = np.hstack([X_num_scaled, X_cat_encoded])
```

### Handling Missing Values
- Numeric features: Filled with 0 (after scaling)
- Categorical features: Encoded as "__missing__"
- Unknown categories at inference: Mapped to -1

## Training Configuration

### Temporal Split
| Split | Seasons | Purpose |
|-------|---------|---------|
| Train | 2014-2021 | Model fitting |
| Validation | 2022-2023 | Hyperparameter tuning |
| Test | 2024-2025 | Final evaluation |

### Why Temporal Split?
- Prevents future data leakage
- Simulates real prediction scenario
- Tests model generalization to new seasons

## Evaluation Metrics

### Regression Metrics
| Metric | Formula | Target |
|--------|---------|--------|
| MAE | mean(\|y - ŷ\|) | < 2.0 positions |
| RMSE | sqrt(mean((y - ŷ)²)) | Minimize |
| Spearman ρ | Rank correlation | > 0.85 |

### Position Bucket Accuracy
| Bucket | Positions | Description |
|--------|-----------|-------------|
| P1-3 | 1, 2, 3 | Podium |
| P4-10 | 4-10 | Points positions |
| P11-20 | 11-20 | Out of points |

### Overfitting Detection
```python
train_val_gap = val_mae - train_mae
# Target: gap < 0.5 positions
```

## Feature Importance Analysis

### Top Features by XGBoost (Typical Results)
| Rank | Feature | Importance |
|------|---------|------------|
| 1 | qualifying_position | ~40% |
| 2 | constructor_prior_wins | ~15% |
| 3 | grid_position | ~8% |
| 4 | driver_avg_positions_gained | ~6% |
| 5 | constructor_prior_points | ~5% |
| 6 | gap_to_teammate_quali | ~4% |
| 7 | driver_form_trend | ~3% |
| 8 | constructor_id | ~3% |
| 9 | driver_circuit_avg_finish | ~3% |
| 10 | driver_prior_position | ~2% |

### Interpretation
- **Qualifying position** is the strongest predictor (~40% importance)
- **Constructor strength** (prior wins, points) is crucial (~20% combined)
- **Driver form** (recent performance, trend) provides signal (~10%)
- **Circuit-specific history** adds context (~3%)

## Inference Pipeline

### Process Flow
```
Input Row ──▶ Load Preprocessor ──▶ Transform Features ──▶ Model Predict ──▶ Output
    │               │                      │                    │              │
    │               ▼                      ▼                    ▼              │
    │         scaler.pkl           X_scaled, X_encoded    model.predict()     │
    │         encoders.pkl                                                     │
    │                                                                          ▼
    └──────────────────────────────────────────────────────────────────────▶ {
                                                                              "predicted_finish_position": 5.2,
                                                                              "podium_probability": 0.15
                                                                            }
```

### Code Example
```python
from app.ml.inference import predict

# Single driver prediction
result = predict(row)
# Returns: {
#   "predicted_finish_position": 5.2,
#   "podium_probability": 0.15
# }
```

## Classification Models

### Podium Prediction
- **Target**: is_podium (1 if position ≤ 3, else 0)
- **Models**: LogisticRegression, RandomForest, GradientBoosting, XGBoost
- **Class Balancing**: class_weight="balanced" or scale_pos_weight

### Top-10 Prediction
- **Target**: is_top_10 (1 if position ≤ 10, else 0)
- **Use Case**: Points scoring probability

## Success Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Test MAE | < 2.0 positions | mean_absolute_error(y_test, pred) |
| Spearman ρ | > 0.85 | spearmanr(y_test, pred)[0] |
| No Overfitting | train/val gap < 0.5 | val_mae - train_mae |

## Model Artifacts

### Saved Files (`app/ml/outputs/`)
| File | Contents |
|------|----------|
| model_regression_xgboost.joblib | Best regression model |
| model_regression_random_forest.joblib | RF regressor |
| model_regression_gradient_boosting.joblib | GB regressor |
| model_regression_ridge.joblib | Ridge regressor |
| model_classification_podium_*.joblib | Podium classifiers |
| model_classification_top10_*.joblib | Top-10 classifiers |
| preprocessor.joblib | Scaler, encoders, feature names |
| evaluation_report.json | Training metrics and analysis |

## API Integration

### Endpoint: `/predict`
```
GET /predict?season=2024&round=1&driver_id=max_verstappen
```

### Response
```json
{
  "predicted_finish_position": 1.5,
  "podium_probability": 0.92
}
```

### Endpoint: `/api/predictions/race`
```
GET /api/predictions/race?season=2024&round=1
```

### Response
```json
{
  "season": 2024,
  "round": 1,
  "predictions": [
    {
      "driver_id": "max_verstappen",
      "constructor_id": "red_bull",
      "predicted_finish_position": 1.5,
      "podium_probability": 0.92
    },
    ...
  ]
}
```

## Limitations and Future Work

### Current Limitations
1. **No Real-Time Data**: Predictions based on historical patterns
2. **Weather Uncertainty**: Weather features require FastF1 data
3. **DNF Modeling**: DNFs not explicitly modeled
4. **Driver Changes**: Mid-season driver changes may not be captured

### Future Improvements
1. **Ensemble Model**: Combine top performers with weighted average
2. **Deep Learning**: Neural network for complex interactions
3. **Uncertainty Quantification**: Prediction intervals
4. **Real-Time Updates**: Live race position tracking
5. **Telemetry Integration**: Car performance data

## References

1. FastF1 Documentation: https://docs.fastf1.dev/
2. Ergast API: http://ergast.com/mrd/
3. XGBoost Documentation: https://xgboost.readthedocs.io/
4. scikit-learn Documentation: https://scikit-learn.org/
