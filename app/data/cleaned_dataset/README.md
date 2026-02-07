# Cleaned F1 Dataset (2014–2025)

Analysis-ready, ML-safe CSVs produced from raw Ergast/Jolpi data by `app/scripts/clean_data.py`.

## Rules applied

- **No URLs** or free-text fields; only identifiers, numerics, categoricals, and dates.
- **Column names:** snake_case; consistent IDs: `driver_id`, `constructor_id`, `circuit_id`, `status_id`, `season`, `round`.
- **IDs:** Categorical; values match across files; no replacement of IDs with names.
- **Missing values:** Left empty; indicator columns added where useful (`is_dnf`, `has_sprint`, `is_sprint`).
- **Relational integrity:** Orphan records removed (FKs reference master tables and races).

## Files

| File | Description |
|------|-------------|
| `seasons.csv` | season |
| `circuits.csv` | circuit_id, circuit_name, country, locality, lat, lng |
| `drivers.csv` | driver_id, first_name, last_name, date_of_birth, nationality, permanent_number |
| `constructors.csv` | constructor_id, constructor_name, nationality |
| `status.csv` | status_id, status_text |
| `races.csv` | season, round, race_name, race_date, circuit_id, has_sprint |
| `results.csv` | season, round, driver_id, constructor_id, grid_position, finish_position, points, laps, status_id, is_dnf |
| `qualifying.csv` | season, round, driver_id, constructor_id, qualifying_position |
| `sprint.csv` | Same as results + is_sprint=true |
| `pitstops.csv` | season, round, driver_id, lap, stop_number, duration_seconds |
| `driver_standings.csv` | season, round, driver_id, standing_position, points, wins |
| `constructor_standings.csv` | season, round, constructor_id, standing_position, points, wins |

## Usage

- **Feature engineering / ML:** Join on `season`, `round`, `driver_id`, `constructor_id`, `circuit_id`, `status_id` as needed.
- **Strategy simulation:** Use `pitstops.csv` (lap, stop_number, duration_seconds) and results (laps, status_id, is_dnf).
- **Backend / frontend:** Same IDs and schema; no merging required at this stage.

## Run cleaning

From repo root:

```bash
python -m app.scripts.clean_data
```

Or with custom paths / season filter:

```python
from app.scripts.clean_data import run_cleaning
run_cleaning(raw_dir="app/data/raw_dataset", clean_dir="app/data/cleaned_dataset", filter_races_season_range=(2014, 2025))
```
