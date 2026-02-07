from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv()
from app.config.config import API_BASE_URL
from app.scripts.collect_data import F1DataFetcher

app = FastAPI(title="F1Insight API", description="Race outcome prediction and strategy assistance")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "F1Insight API", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/collect")
def collect_f1_data(start_year: int = 2014, end_year: int = 2025, include_laps: bool = False):
    """
    Collect F1 data from Ergast API and save to CSV files.

    Args:
        start_year: Starting year for data collection (default: 2014)
        end_year: Ending year for data collection (default: 2025)
        include_laps: Include lap-by-lap timing data (WARNING: creates very large files!)

    Returns:
        F1DataFetcher instance after data collection
    """
    start_year = max(2014, min(2030, start_year))
    end_year = max(start_year, min(2030, end_year))
    fetcher = F1DataFetcher(start_year=start_year, end_year=end_year, base_url=API_BASE_URL)
    fetcher.fetch_all_data(include_laps=include_laps)
    return {"status": "ok", "message": f"Data collected for {start_year}-{end_year}", "include_laps": include_laps}


@app.get("/predict")
def predict(
    season: int = Query(..., ge=2014, le=2030),
    round_num: int = Query(..., ge=1, le=25, alias="round"),
    driver_id: str = Query(..., min_length=1),
):
    """
    Predict finish position and podium probability for a driver at a race.
    Uses merged ML dataset to look up features; requires trained models in app/ml/outputs.
    """
    from app.ml.build_dataset import build_merged_dataset
    from app.ml.inference import predict as run_predict
    clean_dir = Path(__file__).resolve().parent / "data" / "cleaned_dataset"
    df = build_merged_dataset(clean_dir)
    mask = (df["season"] == season) & (df["round"] == round_num) & (df["driver_id"].astype(str).str.strip().str.lower() == driver_id.strip().lower())
    if not mask.any():
        raise HTTPException(status_code=404, detail=f"No data for season={season}, round={round_num}, driver_id={driver_id}")
    row = df.loc[mask].iloc[0]
    result = run_predict(row)
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["message"])
    return result


@app.get("/strategy")
def strategy_recommendation(
    predicted_position_mean: float = Query(..., ge=1, le=20),
    predicted_position_std: float = Query(2.0, ge=0.1, le=10),
    pit_loss_sec: float = Query(22.0, ge=15, le=35),
    n_simulations: int = Query(2000, ge=100, le=10000),
):
    """
    Recommend pit strategy using Monte Carlo simulation.
    Input: expected finishing position and uncertainty (std).
    """
    from app.ml.strategy import recommend_strategy
    out = recommend_strategy(
        predicted_position_mean=predicted_position_mean,
        predicted_position_std=predicted_position_std,
        pit_loss_sec=pit_loss_sec,
        n_simulations=n_simulations,
    )
    return out


# --- Historical data for frontend (no ML in controllers) ---
CLEAN_DIR = Path(__file__).resolve().parent / "data" / "cleaned_dataset"


@app.get("/api/seasons")
def list_seasons():
    """List available seasons from cleaned data."""
    import pandas as pd
    p = CLEAN_DIR / "races.csv"
    if not p.exists():
        return {"seasons": []}
    df = pd.read_csv(p)
    seasons = sorted(df["season"].dropna().unique().astype(int).tolist())
    return {"seasons": seasons}


@app.get("/api/races")
def list_races(season: int = Query(..., ge=2014, le=2030)):
    """List races for a season (round, race_name, circuit_id, date)."""
    import pandas as pd
    p = CLEAN_DIR / "races.csv"
    if not p.exists():
        return {"races": []}
    df = pd.read_csv(p)
    df = df[df["season"] == season].sort_values("round")
    races = df[["round", "race_name", "circuit_id", "race_date"]].to_dict(orient="records")
    return {"season": season, "races": races}


@app.get("/api/drivers")
def list_drivers(season: int = Query(None, ge=2014, le=2030)):
    """List drivers (from cleaned drivers or from results for that season)."""
    import pandas as pd
    drivers_path = CLEAN_DIR / "drivers.csv"
    if drivers_path.exists():
        df = pd.read_csv(drivers_path)[["driver_id", "first_name", "last_name"]].drop_duplicates("driver_id")
        drivers = [{"driver_id": r["driver_id"], "name": f"{r['first_name']} {r['last_name']}"} for _, r in df.iterrows()]
    else:
        drivers = []
    if season is not None:
        res_path = CLEAN_DIR / "results.csv"
        if res_path.exists():
            res = pd.read_csv(res_path)
            res = res[res["season"] == season]["driver_id"].unique()
            if drivers:
                drivers = [d for d in drivers if d["driver_id"] in res]
            else:
                drivers = [{"driver_id": d, "name": d} for d in res]
    return {"drivers": drivers}


def _race_exists_in_calendar(season: int, round_num: int) -> bool:
    """Return True if (season, round) exists in cleaned races calendar."""
    import pandas as pd
    p = CLEAN_DIR / "races.csv"
    if not p.exists():
        return False
    df = pd.read_csv(p)
    return ((df["season"] == season) & (df["round"] == round_num)).any()


@app.get("/api/predictions/race")
def predictions_for_race(
    season: int = Query(..., ge=2014, le=2030),
    round_num: int = Query(..., ge=1, le=25, alias="round"),
):
    """
    Get predictions for all drivers in a race (for dashboard).
    Returns 200 with empty predictions when the race is on the calendar but has no result/qualifying data.
    Returns 404 only when the race is not in the calendar.
    """
    from app.ml.build_dataset import build_merged_dataset
    from app.ml.inference import predict as run_predict
    if not _race_exists_in_calendar(season, round_num):
        raise HTTPException(status_code=404, detail=f"Race not found: season={season}, round={round_num}")
    clean_dir = Path(__file__).resolve().parent / "data" / "cleaned_dataset"
    df = build_merged_dataset(clean_dir)
    mask = (df["season"] == season) & (df["round"] == round_num)
    if not mask.any():
        return {
            "season": season,
            "round": round_num,
            "predictions": [],
            "message": "No result/qualifying data for this race in the dataset. Predictions require collected results for this (season, round).",
        }
    subset = df.loc[mask]
    results = []
    for _, row in subset.iterrows():
        out = run_predict(row)
        if "error" in out:
            continue
        results.append({
            "driver_id": row["driver_id"],
            "constructor_id": row.get("constructor_id", ""),
            "predicted_finish_position": out.get("predicted_finish_position"),
            "podium_probability": out.get("podium_probability"),
        })
    results.sort(key=lambda x: x["predicted_finish_position"] or 99)
    return {"season": season, "round": round_num, "predictions": results}
