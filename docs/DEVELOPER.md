# F1Insight — Developer Documentation

## Setup

1. **Clone and enter project**
   ```bash
   cd /path/to/F1-Insight
   ```

2. **Create virtual environment and install dependencies**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Environment variables**
   - Copy `.env.example` to `.env`.
   - Set `API_BASE_URL` for data collection (default in code: `https://api.jolpi.ca/ergast/f1`).

## Workflows

### Data collection and cleaning

- **Collect raw data (2014–2025, no laps):**
  ```bash
  # Option A: API
  curl "http://localhost:8000/collect?start_year=2014&end_year=2025&include_laps=false"

  # Option B: Python
  python -c "
  from app.config.config import API_BASE_URL
  from app.scripts.collect_data import F1DataFetcher
  F1DataFetcher(2014, 2025, base_url=API_BASE_URL).fetch_all_data(include_laps=False)
  "
  ```
- **Clean raw → cleaned_dataset:**
  ```bash
  python -m app.scripts.clean_data
  ```
- **Full pipeline (collect if results missing, then clean, then train):**
  ```bash
  python -m app.scripts.run_pipeline
  ```
  Skip collect if you already have `app/data/raw_dataset/results.csv`:
  ```bash
  python -m app.scripts.run_pipeline --skip-collect
  ```

### Training and inference

- **Train models (requires cleaned data with results.csv):**
  ```bash
  python -m app.ml.train
  ```
  Outputs: `app/ml/outputs/` (models, preprocessor, evaluation_report.json).

- **Run API server:**
  ```bash
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
  ```
  - Docs: http://localhost:8000/docs  
  - Predict: `GET /predict?season=2024&round=1&driver_id=verstappen`  
  - Strategy: `GET /strategy?predicted_position_mean=5&predicted_position_std=2`

## Frontend

```bash
cd frontend
npm install
npm run dev   # http://localhost:5173
```

Set `VITE_API_URL=http://localhost:8000` in `frontend/.env` if the API runs elsewhere. Build: `npm run build`; preview: `npm run preview`.

## Project layout

- `app/` — Application code
  - `config/config.py` — Config and env defaults
  - `data/raw_dataset/` — Raw CSVs from API
  - `data/cleaned_dataset/` — Cleaned CSVs for ML
  - `ml/` — build_dataset, train, inference, strategy
  - `scripts/` — collect_data, clean_data, run_pipeline
  - `main.py` — FastAPI app
- `frontend/` — React + Tailwind (Vite); dashboard pages
- `docs/` — SYSTEM_DOCUMENTATION.md, ARCHITECTURE.md, TECHNICAL.md, DELIVERABLES.md
- `requirements.txt` — Python dependencies

## Training and inference workflow (summary)

1. Ensure `app/data/cleaned_dataset/results.csv` exists (run collect + clean if not).
2. Run `python -m app.ml.train`.
3. Use `GET /predict` and `GET /strategy` or call `app.ml.inference.predict(row)` and `app.ml.strategy.recommend_strategy(...)` from code.
