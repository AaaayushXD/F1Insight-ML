"""
F1Insight – Raw CSV cleaning, normalization, and standardization.

Produces analysis-ready, ML-safe CSVs from raw Ergast/Jolpi outputs.
Preserves relational consistency; does not merge datasets.
"""

import os
import re
import pandas as pd
from pathlib import Path
from typing import Optional

RAW_DIR = "app/data/raw_dataset"
CLEAN_DIR = "app/data/cleaned_dataset"

# Status IDs that mean "Finished" (not DNF). Ergast: 1 = Finished.
FINISHED_STATUS_IDS = {"1", 1}


def _ensure_clean_dir() -> Path:
    path = Path(CLEAN_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_snake(name: str) -> str:
    s = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name)
    return s.lower().replace("-", "_").replace(" ", "_")


def _normalize_id(val) -> str:
    if pd.isna(val):
        return ""
    return str(val).strip().lower()


def _parse_duration_seconds(val) -> Optional[float]:
    """Parse duration string (e.g. '26.898', '1:23.456') to float seconds."""
    if pd.isna(val) or val == "":
        return None
    s = str(val).strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 2:
            try:
                return float(parts[0]) * 60 + float(parts[1])
            except ValueError:
                return None
        if len(parts) == 3:
            try:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            except ValueError:
                return None
    try:
        return float(s)
    except ValueError:
        return None


def _is_dnf(status_id) -> bool:
    if pd.isna(status_id):
        return True
    return str(status_id).strip() not in FINISHED_STATUS_IDS


def clean_seasons(raw_path: Path, clean_path: Path) -> pd.DataFrame:
    path = raw_path / "seasons.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Keep only season; drop url and any other columns
    out = df[["season"]] if "season" in df.columns else df.copy()
    out = out.dropna(how="all")
    out.to_csv(clean_path / "seasons.csv", index=False)
    return out


def clean_circuits(raw_path: Path, clean_path: Path) -> pd.DataFrame:
    path = raw_path / "circuits.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    col_map = {
        "circuitId": "circuit_id",
        "circuitName": "circuit_name",
        "Location_country": "country",
        "Location_locality": "locality",
        "Location_lat": "lat",
        "Location_long": "lng",
    }
    keep = [c for c in col_map if c in df.columns]
    out = df[keep].rename(columns=col_map)
    out["circuit_id"] = out["circuit_id"].astype(str).str.strip().str.lower()
    out.to_csv(clean_path / "circuits.csv", index=False)
    return out


def clean_drivers(raw_path: Path, clean_path: Path) -> pd.DataFrame:
    path = raw_path / "drivers.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    col_map = {
        "driverId": "driver_id",
        "givenName": "first_name",
        "familyName": "last_name",
        "dateOfBirth": "date_of_birth",
        "nationality": "nationality",
        "permanentNumber": "permanent_number",
    }
    keep = [c for c in col_map if c in df.columns]
    out = df[keep].rename(columns=col_map)
    out["driver_id"] = out["driver_id"].astype(str).str.strip().str.lower()
    out.to_csv(clean_path / "drivers.csv", index=False)
    return out


def clean_constructors(raw_path: Path, clean_path: Path) -> pd.DataFrame:
    path = raw_path / "constructors.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    col_map = {
        "constructorId": "constructor_id",
        "name": "constructor_name",
        "nationality": "nationality",
    }
    keep = [c for c in col_map if c in df.columns]
    out = df[keep].rename(columns=col_map)
    out["constructor_id"] = out["constructor_id"].astype(str).str.strip().str.lower()
    out.to_csv(clean_path / "constructors.csv", index=False)
    return out


def clean_status(raw_path: Path, clean_path: Path) -> pd.DataFrame:
    path = raw_path / "status.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    col_map = {"statusId": "status_id", "status": "status_text"}
    keep = [c for c in col_map if c in df.columns]
    out = df[keep].rename(columns=col_map)
    out["status_id"] = out["status_id"].astype(str).str.strip()
    out.to_csv(clean_path / "status.csv", index=False)
    return out


def clean_races(raw_path: Path, clean_path: Path) -> pd.DataFrame:
    path = raw_path / "races.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)

    # Support both Ergast-style and alternative (year, name, circuitId numeric)
    if "season" in df.columns:
        out = pd.DataFrame()
        out["season"] = df["season"].astype(int)
        out["round"] = df["round"].astype(int)
        out["race_name"] = df["raceName"] if "raceName" in df.columns else df.get("name", "")
        out["race_date"] = df["date"]
        out["circuit_id"] = (
            df["Circuit_circuitId"].astype(str).str.strip().str.lower()
            if "Circuit_circuitId" in df.columns
            else df["circuitId"].astype(str).str.strip().str.lower()
        )
    else:
        out = pd.DataFrame()
        out["season"] = df["year"].astype(int)
        out["round"] = df["round"].astype(int)
        out["race_name"] = df["name"]
        out["race_date"] = df["date"]
        out["circuit_id"] = df["circuitId"].astype(str).str.strip().str.lower()

    # has_sprint: True if Sprint_date or Sprint_date present and non-null
    if "Sprint_date" in df.columns:
        out["has_sprint"] = df["Sprint_date"].notna() & (df["Sprint_date"].astype(str).str.strip() != "")
    else:
        out["has_sprint"] = False

    out.to_csv(clean_path / "races.csv", index=False)
    return out


def clean_results(raw_path: Path, clean_path: Path) -> pd.DataFrame:
    path = raw_path / "results.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)

    driver_col = "Driver_driverId" if "Driver_driverId" in df.columns else "driverId"
    constructor_col = "Constructor_constructorId" if "Constructor_constructorId" in df.columns else "constructorId"
    status_col = "Status_statusId" if "Status_statusId" in df.columns else ("statusId" if "statusId" in df.columns else None)
    status_text_col = "status" if "status" in df.columns else None
    pos_col = "position"
    grid_col = "grid"
    season_col = "season" if "season" in df.columns else "year"

    out = pd.DataFrame()
    out["season"] = df[season_col].astype(int)
    out["round"] = df["round"].astype(int)
    out["driver_id"] = df[driver_col].astype(str).str.strip().str.lower()
    out["constructor_id"] = df[constructor_col].astype(str).str.strip().str.lower()
    out["grid_position"] = pd.to_numeric(df[grid_col], errors="coerce")
    out["finish_position"] = pd.to_numeric(df[pos_col], errors="coerce")
    out["points"] = pd.to_numeric(df["points"], errors="coerce")
    out["laps"] = pd.to_numeric(df["laps"], errors="coerce")

    if status_col and status_col in df.columns:
        out["status_id"] = df[status_col].astype(str).str.strip()
    elif status_text_col:
        status_df = pd.read_csv(raw_path / "status.csv")
        text_to_id = dict(zip(status_df["status"].astype(str).str.strip(), status_df["statusId"].astype(str).str.strip()))
        out["status_id"] = df[status_text_col].astype(str).str.strip().map(lambda x: text_to_id.get(x, ""))
    else:
        out["status_id"] = "1"
    out["is_dnf"] = out["status_id"].apply(_is_dnf)

    out.to_csv(clean_path / "results.csv", index=False)
    return out


def clean_qualifying(raw_path: Path, clean_path: Path) -> pd.DataFrame:
    path = raw_path / "qualifying.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)

    driver_col = "Driver_driverId" if "Driver_driverId" in df.columns else "driverId"
    constructor_col = "Constructor_constructorId" if "Constructor_constructorId" in df.columns else "constructorId"
    pos_col = "position"

    out = pd.DataFrame()
    out["season"] = df["season"].astype(int)
    out["round"] = df["round"].astype(int)
    out["driver_id"] = df[driver_col].astype(str).str.strip().str.lower()
    out["constructor_id"] = df[constructor_col].astype(str).str.strip().str.lower()
    out["qualifying_position"] = pd.to_numeric(df[pos_col], errors="coerce")

    out.to_csv(clean_path / "qualifying.csv", index=False)
    return out


def clean_sprint(raw_path: Path, clean_path: Path) -> pd.DataFrame:
    path = raw_path / "sprint.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)

    driver_col = "Driver_driverId" if "Driver_driverId" in df.columns else "driverId"
    constructor_col = "Constructor_constructorId" if "Constructor_constructorId" in df.columns else "constructorId"
    status_col = "Status_statusId" if "Status_statusId" in df.columns else ("statusId" if "statusId" in df.columns else None)
    status_text_col = "status" if "status" in df.columns else None

    out = pd.DataFrame()
    out["season"] = df["season"].astype(int)
    out["round"] = df["round"].astype(int)
    out["driver_id"] = df[driver_col].astype(str).str.strip().str.lower()
    out["constructor_id"] = df[constructor_col].astype(str).str.strip().str.lower()
    out["grid_position"] = pd.to_numeric(df.get("grid", 0), errors="coerce")
    out["finish_position"] = pd.to_numeric(df["position"], errors="coerce")
    out["points"] = pd.to_numeric(df["points"], errors="coerce")
    out["laps"] = pd.to_numeric(df.get("laps", 0), errors="coerce")
    if status_col and status_col in df.columns:
        out["status_id"] = df[status_col].astype(str).str.strip()
    elif status_text_col:
        status_df = pd.read_csv(raw_path / "status.csv")
        text_to_id = dict(zip(status_df["status"].astype(str).str.strip(), status_df["statusId"].astype(str).str.strip()))
        out["status_id"] = df[status_text_col].astype(str).str.strip().map(lambda x: text_to_id.get(x, ""))
    else:
        out["status_id"] = "1"
    out["is_dnf"] = out["status_id"].apply(_is_dnf)
    out["is_sprint"] = True

    out.to_csv(clean_path / "sprint.csv", index=False)
    return out


def clean_pitstops(raw_path: Path, clean_path: Path, races_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Clean pit stops data, handling both Ergast API format and Kaggle/direct DB format."""
    for name in ("pitstops.csv", "pit_stops.csv"):
        path = raw_path / name
        if path.exists():
            break
    else:
        print("  No pitstops file found in raw data")
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        print("  Pitstops file is empty")
        return pd.DataFrame()

    print(f"  Raw pitstops: {len(df)} rows, columns: {list(df.columns)}")

    if "season" in df.columns and "round" in df.columns:
        # Ergast API format with season/round already present
        season = df["season"].astype(int)
        round_ = df["round"].astype(int)
        driver_col = "Driver_driverId" if "Driver_driverId" in df.columns else "driverId"
        driver_id = df[driver_col].astype(str).str.strip().str.lower()
    elif "raceId" in df.columns:
        # Kaggle/direct DB format: raceId, driverId; need races to get season, round
        if races_df is None:
            races_path = raw_path / "races.csv"
            if races_path.exists():
                races_df = pd.read_csv(races_path)
            else:
                print("  Cannot find races.csv to map raceId -> (season, round)")
                return pd.DataFrame()

        rid_col = "raceId" if "raceId" in races_df.columns else "race_id"
        if rid_col not in races_df.columns:
            print(f"  races.csv missing {rid_col} column")
            return pd.DataFrame()

        season_col = "year" if "year" in races_df.columns else "season"
        race_lookup = races_df[[rid_col, season_col, "round"]].drop_duplicates()
        race_lookup = race_lookup.rename(columns={season_col: "season", rid_col: "raceId"})

        df = df.merge(race_lookup, on="raceId", how="left")

        # Filter out rows where merge failed (no matching race)
        df = df.dropna(subset=["season", "round"])
        if df.empty:
            print("  No pitstops matched to races after merge")
            return pd.DataFrame()

        season = df["season"].astype(int)
        round_ = df["round"].astype(int)
        driver_id = df["driverId"].astype(str).str.strip().str.lower()
    else:
        print(f"  Pitstops file has unknown format. Columns: {list(df.columns)}")
        return pd.DataFrame()

    stop_col = "stop"
    lap_col = "lap"

    out = pd.DataFrame()
    out["season"] = season
    out["round"] = round_
    out["driver_id"] = driver_id
    out["lap"] = pd.to_numeric(df[lap_col], errors="coerce")
    out["stop_number"] = pd.to_numeric(df[stop_col], errors="coerce")

    if "duration" in df.columns:
        out["duration_seconds"] = df["duration"].apply(_parse_duration_seconds)
    elif "milliseconds" in df.columns:
        out["duration_seconds"] = pd.to_numeric(df["milliseconds"], errors="coerce") / 1000.0
    else:
        out["duration_seconds"] = None

    # Remove rows with invalid data
    out = out.dropna(subset=["season", "round", "driver_id"])

    print(f"  Cleaned pitstops: {len(out)} rows")
    out.to_csv(clean_path / "pitstops.csv", index=False)
    return out


def clean_driver_standings(raw_path: Path, clean_path: Path) -> pd.DataFrame:
    path = raw_path / "driver_standings.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)

    driver_col = "Driver_driverId" if "Driver_driverId" in df.columns else "driverId"

    out = pd.DataFrame()
    out["season"] = df["season"].astype(int)
    out["round"] = df["round"].astype(int)
    out["driver_id"] = df[driver_col].astype(str).str.strip().str.lower()
    out["standing_position"] = pd.to_numeric(df["position"], errors="coerce")
    out["points"] = pd.to_numeric(df["points"], errors="coerce")
    out["wins"] = pd.to_numeric(df["wins"], errors="coerce")

    out.to_csv(clean_path / "driver_standings.csv", index=False)
    return out


def clean_constructor_standings(raw_path: Path, clean_path: Path) -> pd.DataFrame:
    path = raw_path / "constructor_standings.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)

    constructor_col = "Constructor_constructorId" if "Constructor_constructorId" in df.columns else "constructorId"

    out = pd.DataFrame()
    out["season"] = df["season"].astype(int)
    out["round"] = df["round"].astype(int)
    out["constructor_id"] = df[constructor_col].astype(str).str.strip().str.lower()
    out["standing_position"] = pd.to_numeric(df["position"], errors="coerce")
    out["points"] = pd.to_numeric(df["points"], errors="coerce")
    out["wins"] = pd.to_numeric(df["wins"], errors="coerce")

    out.to_csv(clean_path / "constructor_standings.csv", index=False)
    return out


def apply_foreign_key_filters(clean_path: Path) -> None:
    """Remove orphan records so all FKs reference master/races.

    More lenient filtering for pitstops to avoid losing valid data when
    driver_id formats don't match (e.g., numeric vs string slugs).
    """
    clean_path = Path(clean_path)
    drivers = pd.read_csv(clean_path / "drivers.csv")["driver_id"].astype(str).str.strip().str.lower()
    constructors = pd.read_csv(clean_path / "constructors.csv")["constructor_id"].astype(str).str.strip().str.lower()
    circuits = pd.read_csv(clean_path / "circuits.csv")["circuit_id"].astype(str).str.strip().str.lower()
    statuses = pd.read_csv(clean_path / "status.csv")["status_id"].astype(str).str.strip()
    races = pd.read_csv(clean_path / "races.csv")
    race_keys = set(zip(races["season"].astype(int), races["round"].astype(int)))

    valid_driver = set(drivers)
    valid_constructor = set(constructors)
    valid_circuit = set(circuits)
    valid_status = set(statuses)

    def filter_df(path: str, keep_fn, warn_only: bool = False) -> int:
        p = clean_path / path
        if not p.exists():
            return 0
        df = pd.read_csv(p)
        if df.empty:
            return 0
        before = len(df)
        try:
            mask = keep_fn(df)
            removed = (~mask).sum()
            if warn_only and removed > 0:
                print(f"  {path}: {removed} rows have FK mismatches (kept anyway)")
                return 0
            df = df.loc[mask]
            df.to_csv(p, index=False)
            return before - len(df)
        except Exception as e:
            print(f"  {path}: filter error: {e}")
            return 0

    # results
    n = filter_df(
        "results.csv",
        lambda d: (
            d["driver_id"].astype(str).str.strip().str.lower().isin(valid_driver)
            & d["constructor_id"].astype(str).str.strip().str.lower().isin(valid_constructor)
            & d["status_id"].astype(str).str.strip().isin(valid_status)
            & d.apply(lambda r: (int(r["season"]), int(r["round"])) in race_keys, axis=1)
        ),
    )
    if n:
        print(f"  results: removed {n} orphan rows")

    # qualifying
    n = filter_df(
        "qualifying.csv",
        lambda d: (
            d["driver_id"].astype(str).str.strip().str.lower().isin(valid_driver)
            & d["constructor_id"].astype(str).str.strip().str.lower().isin(valid_constructor)
            & d.apply(lambda r: (int(r["season"]), int(r["round"])) in race_keys, axis=1)
        ),
    )
    if n:
        print(f"  qualifying: removed {n} orphan rows")

    # sprint
    n = filter_df(
        "sprint.csv",
        lambda d: (
            d["driver_id"].astype(str).str.strip().str.lower().isin(valid_driver)
            & d["constructor_id"].astype(str).str.strip().str.lower().isin(valid_constructor)
            & d.apply(lambda r: (int(r["season"]), int(r["round"])) in race_keys, axis=1)
        ),
    )
    if n:
        print(f"  sprint: removed {n} orphan rows")

    # pitstops: Only filter by race_keys (season/round), be lenient with driver_id
    # because driver_id might be numeric from one source vs slug from another
    p = clean_path / "pitstops.csv"
    if p.exists():
        df = pd.read_csv(p)
        if not df.empty:
            before = len(df)
            # Only filter by race existence
            race_mask = df.apply(lambda r: (int(r["season"]), int(r["round"])) in race_keys, axis=1)
            driver_mask = df["driver_id"].astype(str).str.strip().str.lower().isin(valid_driver)

            # Report but don't filter on driver_id mismatch
            driver_mismatches = (~driver_mask).sum()
            if driver_mismatches > 0:
                print(f"  pitstops: {driver_mismatches} rows have unmatched driver_id (format mismatch)")

            df = df.loc[race_mask]
            df.to_csv(p, index=False)
            removed = before - len(df)
            if removed > 0:
                print(f"  pitstops: removed {removed} rows (race not in cleaned races)")

    # driver_standings
    n = filter_df(
        "driver_standings.csv",
        lambda d: d["driver_id"].astype(str).str.strip().str.lower().isin(valid_driver),
    )
    if n:
        print(f"  driver_standings: removed {n} orphan rows")

    # constructor_standings
    n = filter_df(
        "constructor_standings.csv",
        lambda d: d["constructor_id"].astype(str).str.strip().str.lower().isin(valid_constructor),
    )
    if n:
        print(f"  constructor_standings: removed {n} orphan rows")

    # races: keep only circuit_id in circuits (skip if no overlap, e.g. numeric vs string ID scheme)
    p = clean_path / "races.csv"
    if p.exists():
        df = pd.read_csv(p)
        before = len(df)
        mask = df["circuit_id"].astype(str).str.strip().str.lower().isin(valid_circuit)
        if mask.any():
            df = df.loc[mask]
            df.to_csv(p, index=False)
            if len(df) < before:
                print(f"  races: removed {before - len(df)} orphan rows (circuit_id not in circuits)")
        else:
            print("  races: no circuit_id match with circuits (ID scheme mismatch?); races left unchanged.")


def clean_tyre_stints(raw_path: Path, clean_path: Path) -> pd.DataFrame:
    """Clean FastF1 tyre stint data."""
    path = raw_path / "tyre_stints.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        return df

    # Normalize compound names
    compound_map = {
        "SOFT": "soft",
        "MEDIUM": "medium",
        "HARD": "hard",
        "INTERMEDIATE": "intermediate",
        "WET": "wet",
        "UNKNOWN": "unknown",
    }

    out = pd.DataFrame()
    out["season"] = df["season"].astype(int)
    out["round"] = df["round"].astype(int)
    out["driver_id"] = df["driver_id"].astype(str).str.strip().str.lower()
    out["stint_number"] = pd.to_numeric(df["stint_number"], errors="coerce")
    out["compound"] = df["compound"].astype(str).str.upper().map(
        lambda x: compound_map.get(x, x.lower())
    )
    out["start_lap"] = pd.to_numeric(df["start_lap"], errors="coerce")
    out["end_lap"] = pd.to_numeric(df["end_lap"], errors="coerce")
    out["tyre_life"] = pd.to_numeric(df["tyre_life"], errors="coerce")
    out["avg_lap_time"] = pd.to_numeric(df["avg_lap_time"], errors="coerce")

    out.to_csv(clean_path / "tyre_stints.csv", index=False)
    print(f"  Cleaned tyre stints: {len(out)} rows")
    return out


def clean_weather(raw_path: Path, clean_path: Path) -> pd.DataFrame:
    """Clean FastF1 weather data."""
    path = raw_path / "weather.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        return df

    out = pd.DataFrame()
    out["season"] = df["season"].astype(int)
    out["round"] = df["round"].astype(int)
    out["track_temp"] = pd.to_numeric(df["track_temp"], errors="coerce")
    out["air_temp"] = pd.to_numeric(df["air_temp"], errors="coerce")
    out["humidity"] = pd.to_numeric(df.get("humidity"), errors="coerce")
    out["rainfall"] = df.get("rainfall", False).fillna(False).astype(bool)
    out["wind_speed"] = pd.to_numeric(df.get("wind_speed"), errors="coerce")

    out.to_csv(clean_path / "weather.csv", index=False)
    print(f"  Cleaned weather: {len(out)} rows")
    return out


def run_cleaning(
    raw_dir: str = RAW_DIR,
    clean_dir: str = CLEAN_DIR,
    filter_races_season_range: Optional[tuple] = None,
) -> None:
    raw_path = Path(raw_dir)
    clean_path = _ensure_clean_dir()
    if clean_dir != CLEAN_DIR:
        clean_path = Path(clean_dir)
        clean_path.mkdir(parents=True, exist_ok=True)

    print("Cleaning master and lookup tables...")
    clean_seasons(raw_path, clean_path)
    circuits_df = clean_circuits(raw_path, clean_path)
    drivers_df = clean_drivers(raw_path, clean_path)
    constructors_df = clean_constructors(raw_path, clean_path)
    clean_status(raw_path, clean_path)

    print("Cleaning races...")
    races_df = clean_races(raw_path, clean_path)
    if filter_races_season_range and len(races_df) and "season" in races_df.columns:
        lo, hi = filter_races_season_range
        races_df = races_df[(races_df["season"] >= lo) & (races_df["season"] <= hi)]
        races_df.to_csv(clean_path / "races.csv", index=False)

    print("Cleaning race-level data...")
    clean_results(raw_path, clean_path)
    clean_qualifying(raw_path, clean_path)
    clean_sprint(raw_path, clean_path)
    clean_pitstops(raw_path, clean_path, races_df=pd.read_csv(raw_path / "races.csv") if (raw_path / "races.csv").exists() else None)
    clean_driver_standings(raw_path, clean_path)
    clean_constructor_standings(raw_path, clean_path)

    print("Cleaning FastF1 data (if available)...")
    clean_tyre_stints(raw_path, clean_path)
    clean_weather(raw_path, clean_path)

    print("Applying foreign-key filters (removing orphans)...")
    apply_foreign_key_filters(clean_path)

    print("Running validation checklist...")
    validate_cleaned(clean_path)

    print(f"Cleaned CSVs written to: {clean_path.absolute()}")


def validate_cleaned(clean_path: Path) -> None:
    """Final validation: no URLs, snake_case columns, IDs consistent."""
    clean_path = Path(clean_path)
    url_pattern = re.compile(r"https?://|wikipedia|\.com|\.org", re.I)
    issues = []

    for p in sorted(clean_path.glob("*.csv")):
        df = pd.read_csv(p, nrows=1000)
        # No URL-like values in any cell
        for col in df.columns:
            if df[col].dtype == object:
                if df[col].astype(str).str.contains(url_pattern, na=False).any():
                    issues.append(f"{p.name}: column '{col}' contains URL-like values")
        # Column names snake_case (allow digits)
        for c in df.columns:
            if not re.match(r"^[a-z][a-z0-9_]*$", c):
                issues.append(f"{p.name}: column '{c}' not snake_case")
        # No column named 'url' or ending in '_url'
        if any("url" in c.lower() for c in df.columns):
            issues.append(f"{p.name}: URL column still present")

    if issues:
        for i in issues:
            print(f"  [CHECK] {i}")
    else:
        print("  No URLs in CSVs; column names snake_case; no url columns.")
    print("  Validation done.")


if __name__ == "__main__":
    run_cleaning(filter_races_season_range=(2014, 2025))
