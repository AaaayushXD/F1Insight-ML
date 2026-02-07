"""
Build merged ML dataset: one row per (season, round, driver_id).
Base = results; join qualifying, races, prior standings, pitstops agg, drivers, constructors, circuits.
No future leakage: prior-round standings only.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

CLEAN_DIR = Path(__file__).resolve().parent.parent / "data" / "cleaned_dataset"


def _load_csv(name: str, alt_name: Optional[str] = None) -> pd.DataFrame:
    path = CLEAN_DIR / f"{name}.csv"
    if not path.exists() and alt_name:
        path = CLEAN_DIR / f"{alt_name}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _prior_round_standings(
    standings: pd.DataFrame,
    id_col: str,
    season: int,
    round_num: int,
) -> pd.DataFrame:
    """Get standings from the most recent round < round_num in the same season, or prior season end."""
    same_season = standings[(standings["season"] == season) & (standings["round"] < round_num)]
    if len(same_season) > 0:
        max_round = same_season["round"].max()
        return standings[(standings["season"] == season) & (standings["round"] == max_round)][
            [id_col, "points", "wins", "standing_position"]
        ].rename(columns={"points": f"{id_col}_prior_points", "wins": f"{id_col}_prior_wins", "standing_position": f"{id_col}_prior_position"})
    prev_season = standings[standings["season"] == season - 1]
    if len(prev_season) > 0:
        max_round = prev_season["round"].max()
        return prev_season[prev_season["round"] == max_round][
            [id_col, "points", "wins", "standing_position"]
        ].rename(columns={"points": f"{id_col}_prior_points", "wins": f"{id_col}_prior_wins", "standing_position": f"{id_col}_prior_position"})
    return pd.DataFrame()


def build_merged_dataset(clean_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Build one row per (season, round, driver_id).
    Base table: results if present, else qualifying (targets will be NaN).
    """
    base_dir = clean_dir or CLEAN_DIR
    if not base_dir.exists():
        raise FileNotFoundError(f"Cleaned data directory not found: {base_dir}")

    results = pd.read_csv(base_dir / "results.csv") if (base_dir / "results.csv").exists() else pd.DataFrame()
    if not results.empty:
        base = results.copy()
        base["_from_results"] = True
    else:
        qualifying = _load_csv("qualifying")
        if qualifying.empty:
            qualifying = pd.read_csv(base_dir / "qualifying.csv")
        if qualifying.empty:
            raise FileNotFoundError("Need either results.csv or qualifying.csv in cleaned_dataset to build ML dataset.")
        base = qualifying[["season", "round", "driver_id", "constructor_id"]].drop_duplicates()
        base["_from_results"] = False

    races = pd.read_csv(base_dir / "races.csv")
    races["circuit_id"] = races["circuit_id"].astype(str).str.strip().str.lower()
    base = base.merge(races, on=["season", "round"], how="left", suffixes=("", "_race"))

    qualifying = pd.read_csv(base_dir / "qualifying.csv") if (base_dir / "qualifying.csv").exists() else pd.DataFrame()
    if not qualifying.empty and "qualifying_position" not in base.columns:
        qualifying = qualifying[["season", "round", "driver_id", "qualifying_position"]].drop_duplicates(
            subset=["season", "round", "driver_id"]
        )
        base = base.merge(qualifying, on=["season", "round", "driver_id"], how="left")

    driver_standings = pd.read_csv(base_dir / "driver_standings.csv")
    constructor_standings = pd.read_csv(base_dir / "constructor_standings.csv")

    def get_prior_driver(season: int, round_num: int) -> pd.DataFrame:
        same = driver_standings[(driver_standings["season"] == season) & (driver_standings["round"] < round_num)]
        if len(same) > 0:
            r = same["round"].max()
            return driver_standings[(driver_standings["season"] == season) & (driver_standings["round"] == r)][
                ["driver_id", "points", "wins", "standing_position"]
            ].rename(columns={"points": "driver_prior_points", "wins": "driver_prior_wins", "standing_position": "driver_prior_position"})
        prev = driver_standings[driver_standings["season"] == season - 1]
        if len(prev) > 0:
            r = prev["round"].max()
            return driver_standings[(driver_standings["season"] == season - 1) & (driver_standings["round"] == r)][
                ["driver_id", "points", "wins", "standing_position"]
            ].rename(columns={"points": "driver_prior_points", "wins": "driver_prior_wins", "standing_position": "driver_prior_position"})
        return pd.DataFrame()

    prior_driver_list = []
    for (s, r), _ in base.groupby(["season", "round"]):
        pdf = get_prior_driver(s, r)
        if not pdf.empty:
            pdf["season"], pdf["round"] = s, r
            prior_driver_list.append(pdf)
    if prior_driver_list:
        prior_driver = pd.concat(prior_driver_list, ignore_index=True)
        base = base.merge(prior_driver, on=["season", "round", "driver_id"], how="left")
    else:
        base["driver_prior_points"] = np.nan
        base["driver_prior_wins"] = np.nan
        base["driver_prior_position"] = np.nan

    def get_prior_constructor(season: int, round_num: int) -> pd.DataFrame:
        same = constructor_standings[(constructor_standings["season"] == season) & (constructor_standings["round"] < round_num)]
        if len(same) > 0:
            r = same["round"].max()
            return constructor_standings[(constructor_standings["season"] == season) & (constructor_standings["round"] == r)][
                ["constructor_id", "points", "wins", "standing_position"]
            ].rename(columns={"points": "constructor_prior_points", "wins": "constructor_prior_wins", "standing_position": "constructor_prior_position"})
        prev = constructor_standings[constructor_standings["season"] == season - 1]
        if len(prev) > 0:
            r = prev["round"].max()
            return constructor_standings[(constructor_standings["season"] == season - 1) & (constructor_standings["round"] == r)][
                ["constructor_id", "points", "wins", "standing_position"]
            ].rename(columns={"points": "constructor_prior_points", "wins": "constructor_prior_wins", "standing_position": "constructor_prior_position"})
        return pd.DataFrame()

    prior_cons_list = []
    for (s, r), _ in base.groupby(["season", "round"]):
        pcf = get_prior_constructor(s, r)
        if not pcf.empty:
            pcf["season"], pcf["round"] = s, r
            prior_cons_list.append(pcf)
    if prior_cons_list:
        prior_cons = pd.concat(prior_cons_list, ignore_index=True)
        base = base.merge(prior_cons, on=["season", "round", "constructor_id"], how="left")
    else:
        base["constructor_prior_points"] = np.nan
        base["constructor_prior_wins"] = np.nan
        base["constructor_prior_position"] = np.nan

    pitstops = pd.read_csv(base_dir / "pitstops.csv")
    if not pitstops.empty and len(pitstops) > 0:
        pit_agg = pitstops.groupby(["season", "round", "driver_id"]).agg(
            total_stops=("stop_number", "count"),
            mean_stop_duration=("duration_seconds", "mean"),
        ).reset_index()
        base = base.merge(pit_agg, on=["season", "round", "driver_id"], how="left")
    else:
        base["total_stops"] = np.nan
        base["mean_stop_duration"] = np.nan

    drivers = pd.read_csv(base_dir / "drivers.csv")[["driver_id", "nationality"]].drop_duplicates("driver_id")
    constructors = pd.read_csv(base_dir / "constructors.csv")[["constructor_id", "nationality"]].rename(columns={"nationality": "constructor_nationality"})
    circuits = pd.read_csv(base_dir / "circuits.csv")[["circuit_id", "lat", "lng"]].rename(columns={"lat": "circuit_lat", "lng": "circuit_lng"})
    circuits["circuit_id"] = circuits["circuit_id"].astype(str).str.strip().str.lower()

    base = base.merge(drivers, on="driver_id", how="left")
    base = base.merge(constructors, on="constructor_id", how="left")
    base = base.merge(circuits, on="circuit_id", how="left")

    if not results.empty:
        if "grid_position" not in base.columns and "grid" in results.columns:
            base["grid_position"] = results["grid_position"]
        if "finish_position" not in base.columns:
            base["finish_position"] = results["position"] if "position" in results.columns else np.nan
        if "points" not in base.columns:
            base["points"] = results["points"]
        if "laps" not in base.columns:
            base["laps"] = results["laps"]
        if "status_id" in results.columns:
            base["status_id"] = results["status_id"]
        if "is_dnf" not in base.columns and "status_id" in base.columns:
            base["is_dnf"] = base["status_id"].astype(str).str.strip() != "1"
    else:
        base["finish_position"] = np.nan
        base["grid_position"] = base.get("grid_position", np.nan)
        base["points"] = np.nan
        base["laps"] = np.nan
        base["is_dnf"] = np.nan

    base["is_podium"] = (base["finish_position"].notna() & (base["finish_position"] <= 3)).astype(float)
    base.loc[base["finish_position"].isna(), "is_podium"] = np.nan
    base["is_top_10"] = (base["finish_position"].notna() & (base["finish_position"] <= 10)).astype(float)
    base.loc[base["finish_position"].isna(), "is_top_10"] = np.nan

    # --- Rolling and circuit features (no future leakage: only past races) ---
    if not results.empty and "finish_position" in results.columns and "grid_position" in results.columns:
        res = results[["season", "round", "driver_id", "constructor_id", "finish_position", "grid_position"]].copy()
        res["_ord"] = res["season"] * 1000 + res["round"]
        res = res.sort_values(["_ord"])
        res["positions_gained"] = res["grid_position"] - res["finish_position"]  # positive = gained

        races_df = pd.read_csv(base_dir / "races.csv")[["season", "round", "circuit_id"]]
        races_df["circuit_id"] = races_df["circuit_id"].astype(str).str.strip().str.lower()
        res = res.merge(races_df, on=["season", "round"], how="left")

        roll_n = 5
        driver_recent = []
        driver_circuit = []
        driver_gain = []
        cons_recent = []
        for _, row in base.iterrows():
            s, r, did, cid, cid_circ = row["season"], row["round"], row["driver_id"], row["constructor_id"], row.get("circuit_id", "")
            ord_cur = s * 1000 + r
            past = res[res["_ord"] < ord_cur]
            dr_past = past[past["driver_id"] == did].tail(roll_n)
            dc_past = past[(past["driver_id"] == did) & (past["circuit_id"] == cid_circ)]
            cons_past = past[past["constructor_id"] == cid].tail(roll_n)
            driver_recent.append(dr_past["finish_position"].mean() if len(dr_past) else np.nan)
            driver_circuit.append(dc_past["finish_position"].mean() if len(dc_past) else np.nan)
            driver_gain.append(dr_past["positions_gained"].mean() if len(dr_past) else np.nan)
            cons_recent.append(cons_past["finish_position"].mean() if len(cons_past) else np.nan)
        base["driver_recent_avg_finish"] = driver_recent
        base["driver_circuit_avg_finish"] = driver_circuit
        base["driver_avg_positions_gained"] = driver_gain
        base["constructor_recent_avg_finish"] = cons_recent
    else:
        base["driver_recent_avg_finish"] = np.nan
        base["driver_circuit_avg_finish"] = np.nan
        base["driver_avg_positions_gained"] = np.nan
        base["constructor_recent_avg_finish"] = np.nan

    return base


if __name__ == "__main__":
    df = build_merged_dataset()
    print(df.shape)
    print(df.columns.tolist())
    print(df.head(2))
