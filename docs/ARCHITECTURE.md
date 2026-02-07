# F1Insight — System Architecture & Phase 1 Checklist

**Project:** F1Insight – Race Outcome Prediction & Intelligent Strategy Assistance System  
**Scope:** 2014–2025 F1 data (Ergast/Jolpi API), ML models, strategy simulation, FastAPI backend.

---

## 1. Architecture Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           F1Insight System                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  DATA LAYER                                                                   │
│  ┌──────────────┐    ┌─────────────────┐    ┌────────────────────────────┐  │
│  │ Ergast/Jolpi │───▶│ raw_dataset/    │───▶│ cleaned_dataset/           │  │
│  │ API          │    │ (collect_data)  │    │ (clean_data)               │  │
│  └──────────────┘    └─────────────────┘    └──────────────┬─────────────┘  │
│         ▲                           ▲                       │                │
│         │                           │                       ▼                │
│  ┌──────┴──────┐            ┌──────┴──────┐    ┌────────────────────────────┐  │
│  │ GET /collect │            │ clean_data │    │ build_dataset.py           │  │
│  │ (FastAPI)    │            │ (script)   │    │ merged ML-ready table      │  │
│  └─────────────┘            └────────────┘    └──────────────┬─────────────┘  │
│                                                               │                │
├───────────────────────────────────────────────────────────────┼────────────────┤
│  ML LAYER                                                     ▼                │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │ train.py                                                                 │  │
│  │ • Temporal split: train 2014–2021, val 2022–2023, test 2024–2025          │  │
│  │ • Regression: finish_position (Ridge, RF, XGBoost)                       │  │
│  │ • Classification: is_podium, is_top_10 (Logistic, RF, XGBoost)          │  │
│  │ • Outputs: app/ml/outputs/ (models, preprocessor, evaluation_report.json)│  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────────┤
│  STRATEGY LAYER (strategy.py: clustering + Monte Carlo)                                           │
│  • Tyre strategy clustering / pit-stop window estimation                     │  │
│  • Monte Carlo simulation for strategy ranking                                │  │
├─────────────────────────────────────────────────────────────────────────────┤
│  API LAYER                                                                    │
│  • FastAPI app (main.py): /, /health, /collect                                │  │
│  • /predict, /strategy, /api/seasons, /api/races, /api/drivers, /api/predictions/race                       │  │
├─────────────────────────────────────────────────────────────────────────────┤
│  FRONTEND (frontend/: React + Tailwind)                                                       │
│  • Home, Race predictions, Strategy, Driver compare          │  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. What Exists vs Proposed

| Component | Status | Notes |
|-----------|--------|--------|
| Data ingestion (Ergast 2014–2025) | ✅ Exists | `F1DataFetcher` in `collect_data.py`; requires `API_BASE_URL` in env |
| Raw CSV persistence | ✅ Exists | 12 entity types; `results.csv` / `sprint.csv` must be produced by running collector |
| Cleaning & normalization | ✅ Exists | `clean_data.py`; snake_case, FK filters, DNF handling |
| Merged ML dataset | ✅ Exists | `build_dataset.py`; prior-round standings, no future leakage |
| Regression (position) | ✅ Exists | Ridge, RandomForest, XGBoost in `train.py` |
| Classification (podium, top-10) | ✅ Exists | Logistic, RF, XGBoost; is_dnf not yet trained |
| Gradient Boosting (from scratch) | ⚠️ Partial | sklearn RF/XGB used; “from scratch” GBM can be added as optional |
| Strategy (clustering + Monte Carlo) | ✅ Exists | strategy.py; degradation/traffic in MC |
| Backend REST APIs | ✅ Exists | /predict, /strategy, /api/* for frontend |
| Frontend (React + Tailwind) | ✅ Exists | frontend/ with predictions, strategy, compare |
| RL | ❌ Not required | Explicitly optional; excluded is acceptable |

---

## 3. Alignment with Synopsis / Objectives

- **Race outcome prediction:** Implemented via regression (finish position) and classification (podium, top-10).  
- **Intelligent strategy assistance:** Not yet implemented (clustering + Monte Carlo planned).  
- **Data sources:** Ergast (Jolpi) 2014–2025 mandatory and used; FastF1/Kaggle optional and not integrated.  
- **Scope boundaries:** No over-engineering; strategy and prediction APIs are under-implementation.

---

## 4. Issues Identified

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| `results.csv` / `sprint.csv` missing in raw/cleaned | High | data/ | Run data collection; add pipeline script |
| `/collect` ignores `include_laps` query param | Medium | main.py | Use `include_laps=include_laps` in `fetch_all_data()` |
| Config empty; API_BASE_URL required at import | Medium | config, collect_data | Add default in config; document .env |
| Standings “prior round” may be end-of-season only | Low | build_dataset | Document; accept or add per-round if API allows |
| Strategy module absent | High | ml/ or app/ | Implement clustering + Monte Carlo |
| Prediction/strategy APIs absent | High | main.py | Add /predict, /strategy endpoints |
| is_dnf classification not trained | Low | train.py | Add is_dnf target in training loop |
| No GradientBoostingRegressor/Classifier (sklearn) | Low | train.py | Add GBM for parity with “Gradient Boosting” requirement |

---

## 5. Data Leakage & Validation

- **Temporal split:** Train 2014–2021, val 2022–2023, test 2024–2025 — no future leakage.  
- **Prior standings:** Only standings from rounds &lt; current round (or previous season end) used — correct.  
- **Pit stops:** Aggregated per (season, round, driver); used as feature (post-race aggregate). For true pre-race prediction, pit-stop features would need to be lagged or excluded from position model; currently acceptable for “race outcome” as historical pattern.  
- **Targets:** finish_position, is_podium, is_top_10 from results — no leakage.

---

## 6. Security & API Design

- No secrets in code; `API_BASE_URL` from env.  
- `/collect` is heavy and blocking; consider rate limiting and background task in production.  
- Input validation: `/collect` accepts start_year, end_year, include_laps — validate ranges (e.g. 2014–2025).

---

## 7. Next Steps (Phases 2–8)

1. **Phase 2:** Ensure results/sprint collected; fix main.py; validate cleaning.  
2. **Phase 3:** Add GradientBoosting; improve features; full metrics (e.g. NDCG if applicable).  
3. **Phase 4:** Implement strategy module (clustering + Monte Carlo).  
4. **Phase 5:** Add /predict and /strategy APIs; optional React frontend.  
5. **Phase 6:** Code review; refactor; input validation.  
6. **Phase 7:** Technical and developer documentation.  
7. **Phase 8:** End-to-end test; supervisor-ready runbook.
