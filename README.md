# F1Insight — Race Outcome Prediction & Intelligent Strategy Assistance

Final-year academic project: predict F1 race outcomes and recommend pit-stop strategy using ML and simulation.

## Features

- **Data:** Ergast-compatible API (2014–2025); raw → cleaned → merged ML dataset with prior-round standings (no leakage).
- **Models:** Regression (finish position): Ridge, Random Forest, Gradient Boosting, XGBoost. Classification (podium, top-10): Logistic, RF, GBM, XGBoost.
- **Strategy:** Pit-stop strategy clustering (KMeans) and Monte Carlo simulation to rank strategies by expected position.
- **API:** FastAPI — `/collect`, `/predict`, `/strategy`, `/health`.

## Quick start

```bash
# Backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set API_BASE_URL if you will collect data
python -m app.scripts.clean_data   # after raw data exists
python -m app.ml.train
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend (new terminal)
cd frontend && npm install && npm run dev
# Open http://localhost:5173 — API base: http://localhost:8000 (set VITE_API_URL if different)
```

## Project structure

| Path | Purpose |
|------|--------|
| `app/scripts/collect_data.py` | Fetch F1 data from API → raw_dataset |
| `app/scripts/clean_data.py` | Clean and normalize → cleaned_dataset |
| `app/ml/build_dataset.py` | Build merged ML table (no future leakage) |
| `app/ml/train.py` | Train regression and classification models |
| `app/ml/strategy.py` | Strategy clustering and Monte Carlo |
| `app/ml/inference.py` | Load models and predict |
| `app/main.py` | FastAPI app, CORS, all routes |
| `frontend/` | React + Tailwind dashboard (predictions, strategy, compare) |
| `docs/SYSTEM_DOCUMENTATION.md` | Full system and deployment |
| `docs/ARCHITECTURE.md` | Architecture and phase checklist |
| `docs/TECHNICAL.md` | Technical design and limitations |
| `docs/DEVELOPER.md` | Setup and workflows |

## API summary

- **GET /predict?season=2024&round=1&driver_id=verstappen** — Predicted position and podium probability (requires trained models).
- **GET /strategy?predicted_position_mean=5&predicted_position_std=2** — Best strategy and ranking (Monte Carlo).
- **GET /collect?start_year=2014&end_year=2025** — Run data collection (blocking).

## Academic alignment

- **Objectives:** Race outcome prediction (regression + classification) and strategy assistance (clustering + simulation) are implemented and documented.
- **Scope:** 2014–2025 data; Ergast mandatory; FastF1/Kaggle optional and not integrated; RL optional and excluded.
- **Deliverables:** Data report (`docs/data-collection-and-dataset-analysis-report.md`), architecture and technical docs, evaluation report in `app/ml/outputs/evaluation_report.json` after training.

## License and attribution

Project for academic use. Data from Ergast-compatible API (e.g. Jolpi).
