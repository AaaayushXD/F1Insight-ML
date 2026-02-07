# F1Insight — System Architecture & Implementation Documentation

**Project:** F1Insight – Race Outcome Prediction & Intelligent Strategy Assistance System  
**Academic final-year project. Data: Ergast API 2014–2025. No real-time telemetry. No proprietary data.**

---

## 1. System architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  FRONTEND (React + Tailwind)  —  http://localhost:5173                           │
│  • Home, Race predictions, Strategy, Driver compare                               │
│  • Calls backend via VITE_API_URL (default http://localhost:8000)                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  BACKEND (FastAPI)  —  http://localhost:8000                                     │
│  • GET /api/seasons, /api/races, /api/drivers, /api/predictions/race              │
│  • GET /predict, /strategy, /collect, /health                                    │
│  • CORS enabled for frontend. No ML logic in route handlers; uses ml.inference   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
          ┌─────────────────────────────┼─────────────────────────────┐
          ▼                             ▼                             ▼
┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│  Data layer      │         │  ML layer        │         │  Strategy layer │
│  raw_dataset/    │         │  build_dataset   │         │  strategy.py    │
│  cleaned_dataset/│         │  train.py        │         │  clustering +   │
│  (Ergast CSV)    │         │  inference.py    │         │  Monte Carlo    │
└──────────────────┘         └──────────────────┘         └──────────────────┘
```

---

## 2. Data pipeline

- **Ingestion:** `app/scripts/collect_data.py` (F1DataFetcher). Ergast-compatible API (e.g. Jolpi). Writes: seasons, circuits, constructors, drivers, status, races, results, qualifying, sprint, pitstops, driver_standings, constructor_standings.
- **Cleaning:** `app/scripts/clean_data.py`. Snake_case columns, normalized IDs, DNF flag from status (text or statusId), FK filters. Output: `app/data/cleaned_dataset/`.
- **ML dataset:** `app/ml/build_dataset.py`. One row per (season, round, driver_id). Joins: results, qualifying, races, **prior-round** driver/constructor standings (no future leakage), pit-stop aggregates, driver/circuit/constructor lookups. **Engineered features:** driver_recent_avg_finish, driver_circuit_avg_finish, driver_avg_positions_gained, constructor_recent_avg_finish (all from past races only).
- **Splits:** Train 2014–2021, val 2022–2023, test 2024–2025. No shuffle across seasons.

---

## 3. Model design & evaluation

- **Regression (finish position):** Ridge, Random Forest, Gradient Boosting, XGBoost. Metrics: MAE, RMSE, Spearman (when val/test available). Best model chosen by lowest MAE; justification written to evaluation_report.json.
- **Classification (podium, top-10):** Logistic, RF, GBM, XGBoost. Metrics: accuracy, F1, ROC-AUC. Podium classifier used for /predict podium_probability.
- **Preprocessing:** StandardScaler (numeric), LabelEncoder (categorical). Preprocessor and feature list saved with models. Unknown categories at inference mapped to -1.
- **Artifacts:** `app/ml/outputs/` — preprocessor.joblib, model_regression_*.joblib, model_classification_podium_*.joblib, evaluation_report.json, training_results.png.

---

## 4. Strategy logic

- **Clustering:** Historical pit stops aggregated per (season, round, driver_id). KMeans(n=3) on (num_stops, mean_lap_of_stop, mean_duration_seconds). Produces strategy labels (e.g. one-stop, two-stop).
- **Monte Carlo:** Simulates N outcomes from N(predicted_position_mean, predicted_position_std). Adds **degradation_std** and **traffic_loss_std** (no telemetry; configurable uncertainty). Per-strategy penalty from extra pit stops (pit_loss_sec → position penalty). Strategies ranked by expected finishing position.
- **Recommendation:** `recommend_strategy()` returns best strategy and full ranking. Strategy depends on ML-predicted position (user or API supplies mean/std).

---

## 5. API design

| Endpoint | Purpose |
|----------|---------|
| GET /api/seasons | List available seasons (for UI) |
| GET /api/races?season= | List races for season |
| GET /api/drivers?season= | List drivers (optionally for season) |
| GET /api/predictions/race?season=&round= | Predictions for all drivers in a race |
| GET /predict?season=&round=&driver_id= | Single-driver prediction |
| GET /strategy?predicted_position_mean=&predicted_position_std= | Strategy recommendation |
| GET /collect | Trigger data collection (admin) |
| GET /health | Health check |

Input validation on query params. Errors return 4xx/5xx with detail. No ML logic in route handlers; they call ml.inference and ml.strategy.

---

## 6. Frontend

- **Stack:** Vite, React 18, React Router, Tailwind CSS.
- **Pages:** Home (overview), Race predictions (season/race select, table of predictions), Strategy (mean/std inputs, best strategy + ranking), Driver compare (two drivers, same race, predictions + strategy).
- **API base:** `VITE_API_URL` (default http://localhost:8000). CORS allowed from backend for localhost:5173 and 3000.
- **Design:** Primary #2C64DD, accent #A1BAF0, surface #E8EEFF, satellite #5E3FEF, H2 #3C3C43.

---

## 7. Limitations & assumptions

- **No telemetry:** Strategy uses historical aggregates and parametric uncertainty (degradation_std, traffic_loss_std), not live tyre wear.
- **No hardcoded results:** All targets from cleaned results.csv; predictions from trained models only.
- **Data leakage:** Prevented by prior-round standings and past-only rolling/circuit features.
- **Explainability:** Feature importance in report; strategy ranking is interpretable (expected position per strategy).
- **API data scope:** Collector fetches year-by-year; some APIs may paginate. Full 2014–2025 coverage may require multiple runs or pagination support.

---

## 8. Setup & deployment

- **Backend:** `python -m venv venv && source venv/bin/activate`, `pip install -r requirements.txt`, set `API_BASE_URL` in .env if collecting. Run: `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`.
- **Frontend:** `cd frontend && npm install && npm run dev` (dev) or `npm run build && npm run preview` (preview build). Set `VITE_API_URL` if backend is not on localhost:8000.
- **Training:** After cleaning data, `python -m app.ml.train`. Optionally `python -m app.ml.plot_training_results` for plots.
- **Full pipeline:** `python -m app.scripts.run_pipeline` (optional collect → clean → train).

---

## 9. File map

| Path | Role |
|------|------|
| app/main.py | FastAPI app, CORS, routes |
| app/config/config.py | API_BASE_URL, paths, split constants |
| app/scripts/collect_data.py | Ergast ingestion |
| app/scripts/clean_data.py | Cleaning, status mapping |
| app/scripts/run_pipeline.py | collect → clean → train |
| app/ml/build_dataset.py | Merged dataset, rolling/circuit features |
| app/ml/train.py | Train/eval, best-model selection |
| app/ml/inference.py | Load models, predict |
| app/ml/strategy.py | Clustering, Monte Carlo, recommend |
| app/ml/plot_training_results.py | Matplotlib plots from report |
| frontend/src/App.jsx | Router, nav |
| frontend/src/pages/*.jsx | Home, RacePredictions, Strategy, DriverCompare |

This document reflects the implemented system and is suitable for viva and submission.
