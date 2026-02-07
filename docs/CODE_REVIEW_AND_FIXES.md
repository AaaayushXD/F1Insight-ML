# F1Insight — Code Review and Applied Fixes

## Issues identified and fixed

| # | Issue | Fix |
|---|--------|-----|
| 1 | `/collect` ignored `include_laps` query param | Use `include_laps=include_laps` in `fetch_all_data()` and return JSON instead of fetcher object. |
| 2 | `API_BASE_URL` required at class definition (app failed to start without .env) | Moved URL resolution to `F1DataFetcher.__init__`; added optional `base_url` and config default in `app/config/config.py`. |
| 3 | No input validation on `/collect` year range | Clamp `start_year`/`end_year` to 2014–2030 in `main.py`. |
| 4 | Gradient Boosting (from scratch/sklearn) not in training | Added `GradientBoostingRegressor` and `GradientBoostingClassifier` in `train.py`. |
| 5 | Only podium classification trained; top-10 not | Added full `is_top_10` classification block with same model set and saved models. |
| 6 | Strategy module missing | Implemented `app/ml/strategy.py`: pit-stop clustering (KMeans) and Monte Carlo strategy ranking. |
| 7 | No prediction or strategy APIs | Added `GET /predict` (season, round, driver_id) and `GET /strategy` (predicted_position_mean, std, etc.) in `main.py`. |
| 8 | No inference entry point for API | Added `app/ml/inference.py`: load preprocessor and models, single-row prediction. |
| 9 | Empty `app/config/config.py` | Added `API_BASE_URL`, dataset dirs, and ML split constants. |
| 10 | No pipeline script to run collect → clean → train | Added `app/scripts/run_pipeline.py` with `--skip-collect`, `--collect-only`, `--clean-only`, `--train-only`. |
| 11 | Missing `.env.example` | Added with `API_BASE_URL` and optional paths. |
| 12 | `round` as query param name shadows builtin | Used `round_num` with `alias="round"` in FastAPI. |

## Security and reliability

- No secrets in code; `API_BASE_URL` from env or config default.
- Query params validated (ranges, min_length where applicable).
- `/collect` is blocking and heavy; consider background task and rate limiting for production.
- Strategy and predict endpoints are read-only and do not expose internal paths.

## Naming and structure

- Consistent snake_case in Python; clear separation: scripts (data), ml (models, inference, strategy), config, main.
- No circular imports; inference imports train only for `_apply_preprocessor`.

## Performance

- Training loads full merged dataset once; temporal split in memory.
- Inference loads artifacts once per process; no per-request model reload.
- Monte Carlo uses fixed RNG seed for reproducibility.
