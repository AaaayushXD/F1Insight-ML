# F1Insight — Technical Documentation

## System Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the high-level map and phase checklist.

### Data Pipeline

1. **Ingestion** (`app/scripts/collect_data.py`): F1DataFetcher calls Ergast-compatible API (2014–2025), paginates and flattens JSON, writes one CSV per entity to `app/data/raw_dataset/`.
2. **Cleaning** (`app/scripts/clean_data.py`): Normalizes column names (snake_case), standardizes IDs, parses durations, adds `is_dnf`, applies FK filters. Output: `app/data/cleaned_dataset/`.
3. **ML dataset** (`app/ml/build_dataset.py`): Joins results → qualifying, races, **prior-round** driver/constructor standings (no future leakage), pit-stop aggregates, circuits/drivers/constructors. One row per (season, round, driver_id). Targets: `finish_position`, `is_podium`, `is_top_10`, `is_dnf`.

### Model Design

- **Temporal split:** Train 2014–2021, validation 2022–2023, test 2024–2025.
- **Regression:** Predict `finish_position` (MAE, RMSE, Spearman). Models: Ridge, Random Forest, Gradient Boosting, XGBoost.
- **Classification:** Predict `is_podium` and `is_top_10` (accuracy, F1, ROC-AUC). Same model families plus Logistic Regression.
- **Preprocessing:** StandardScaler on numeric features; LabelEncoder on categoricals (driver_id, constructor_id, circuit_id). Unknown categories at inference get encoder index -1.
- **Artifacts:** `app/ml/outputs/` stores `preprocessor.joblib`, `model_regression_*.joblib`, `model_classification_podium_*.joblib`, `model_classification_top10_*.joblib`, `evaluation_report.json`.

### Strategy Logic

- **Clustering** (`app/ml/strategy.py`): Historical pit stops aggregated per (season, round, driver_id). KMeans (n_clusters=3) on (num_stops, mean_lap_of_stop, mean_duration_seconds) to get strategy types.
- **Monte Carlo:** For a given predicted position (mean ± std), simulate N outcomes. Optionally add position penalty per extra pit stop (pit_loss_sec converted to position loss). Rank strategies by expected finishing position.
- **Recommendation:** `recommend_strategy(predicted_position_mean, predicted_position_std, ...)` returns best strategy and full ranking.

### API Design

- **GET /collect:** Triggers data collection (start_year, end_year, include_laps). Blocking; use for admin/setup.
- **GET /predict:** season, round, driver_id → predicted_finish_position, podium_probability (if classifier present). Looks up merged row and runs inference.
- **GET /strategy:** predicted_position_mean, predicted_position_std, pit_loss_sec, n_simulations → best_strategy and strategy_ranking.

### Assumptions and Limitations

- **Data:** Ergast/Jolpi only; no telemetry or per-lap weather. Standings may be end-of-season only per year (prior-round form can be same for all rounds in a season).
- **Strategy:** Pit-stop clustering uses historical aggregates; Monte Carlo uses a simple position penalty for extra stops. No tyre compound or degradation model.
- **RL:** Not implemented; excluded by design for this scope.
