"""
FastF1 data collector for enhanced F1 race data.

Collects:
- Tyre stint information (compound, lap counts, degradation)
- Weather data (track temp, air temp, humidity, rain)
- Lap times for performance analysis

FastF1 provides reliable data from 2018 onwards.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

# Suppress FastF1 warnings during import
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import fastf1
    from fastf1 import get_session
    HAS_FASTF1 = True
except ImportError:
    HAS_FASTF1 = False
    print("FastF1 not installed. Install with: pip install fastf1", file=sys.stderr)

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.config.config import (
    FASTF1_CACHE_DIR,
    FASTF1_START_YEAR,
    FASTF1_END_YEAR,
    RAW_DATASET_DIR,
)


class FastF1Collector:
    """Collect enhanced F1 data from FastF1 library."""

    def __init__(
        self,
        start_year: int = FASTF1_START_YEAR,
        end_year: int = FASTF1_END_YEAR,
        cache_dir: str = FASTF1_CACHE_DIR,
        output_dir: str = RAW_DATASET_DIR,
    ):
        if not HAS_FASTF1:
            raise ImportError("FastF1 is required. Install with: pip install fastf1")

        self.start_year = start_year
        self.end_year = end_year
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)

        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Enable FastF1 cache
        fastf1.Cache.enable_cache(str(self.cache_dir))

    def get_event_schedule(self, year: int) -> pd.DataFrame:
        """Get the race schedule for a given year."""
        try:
            schedule = fastf1.get_event_schedule(year)
            return schedule
        except Exception as e:
            print(f"Error getting schedule for {year}: {e}")
            return pd.DataFrame()

    def get_session_safely(
        self, year: int, round_num: int, session_type: str = "R"
    ) -> Optional[Any]:
        """Safely load a FastF1 session with error handling."""
        try:
            session = get_session(year, round_num, session_type)
            session.load(
                laps=True,
                telemetry=False,  # Skip telemetry to save time/memory
                weather=True,
                messages=False,
            )
            return session
        except Exception as e:
            print(f"  Error loading session {year} R{round_num}: {e}")
            return None

    def extract_tyre_stints(self, session) -> List[Dict]:
        """Extract tyre stint data from a session."""
        stints = []
        try:
            laps = session.laps
            if laps is None or laps.empty:
                return stints

            for driver in laps["Driver"].unique():
                driver_laps = laps[laps["Driver"] == driver].copy()
                if driver_laps.empty:
                    continue

                # Get driver ID (lowercase slug format)
                driver_info = session.get_driver(driver)
                driver_id = driver_info.get("Abbreviation", driver).lower()

                # Group laps by stint
                driver_laps = driver_laps.sort_values("LapNumber")
                stint_num = 0
                prev_compound = None
                stint_start = 1

                for idx, lap in driver_laps.iterrows():
                    compound = lap.get("Compound", "UNKNOWN")
                    lap_num = lap.get("LapNumber", 0)
                    lap_time = lap.get("LapTime")

                    if compound != prev_compound and prev_compound is not None:
                        # End of previous stint
                        stint_laps = driver_laps[
                            (driver_laps["LapNumber"] >= stint_start)
                            & (driver_laps["LapNumber"] < lap_num)
                        ]
                        avg_lap_time = None
                        if not stint_laps.empty:
                            valid_times = stint_laps["LapTime"].dropna()
                            if len(valid_times) > 0:
                                avg_lap_time = valid_times.mean().total_seconds()

                        stints.append({
                            "season": session.event["EventDate"].year,
                            "round": session.event["RoundNumber"],
                            "driver_id": driver_id,
                            "stint_number": stint_num,
                            "compound": prev_compound,
                            "start_lap": stint_start,
                            "end_lap": lap_num - 1,
                            "tyre_life": lap_num - stint_start,
                            "avg_lap_time": avg_lap_time,
                        })
                        stint_num += 1
                        stint_start = lap_num

                    prev_compound = compound

                # Final stint
                if prev_compound is not None:
                    last_lap = driver_laps["LapNumber"].max()
                    stint_laps = driver_laps[driver_laps["LapNumber"] >= stint_start]
                    avg_lap_time = None
                    if not stint_laps.empty:
                        valid_times = stint_laps["LapTime"].dropna()
                        if len(valid_times) > 0:
                            avg_lap_time = valid_times.mean().total_seconds()

                    stints.append({
                        "season": session.event["EventDate"].year,
                        "round": session.event["RoundNumber"],
                        "driver_id": driver_id,
                        "stint_number": stint_num,
                        "compound": prev_compound,
                        "start_lap": stint_start,
                        "end_lap": int(last_lap),
                        "tyre_life": int(last_lap - stint_start + 1),
                        "avg_lap_time": avg_lap_time,
                    })

        except Exception as e:
            print(f"  Error extracting tyre stints: {e}")

        return stints

    def extract_weather(self, session) -> Optional[Dict]:
        """Extract weather data from a session."""
        try:
            weather = session.weather_data
            if weather is None or weather.empty:
                return None

            # Get average weather conditions during the race
            return {
                "season": session.event["EventDate"].year,
                "round": session.event["RoundNumber"],
                "track_temp": weather["TrackTemp"].mean(),
                "air_temp": weather["AirTemp"].mean(),
                "humidity": weather["Humidity"].mean() if "Humidity" in weather.columns else None,
                "rainfall": weather["Rainfall"].any() if "Rainfall" in weather.columns else False,
                "wind_speed": weather["WindSpeed"].mean() if "WindSpeed" in weather.columns else None,
            }
        except Exception as e:
            print(f"  Error extracting weather: {e}")
            return None

    def extract_driver_results(self, session) -> List[Dict]:
        """Extract driver results with additional FastF1 data."""
        results = []
        try:
            session_results = session.results
            if session_results is None or session_results.empty:
                return results

            for _, row in session_results.iterrows():
                driver_id = row.get("Abbreviation", "").lower()
                if not driver_id:
                    continue

                results.append({
                    "season": session.event["EventDate"].year,
                    "round": session.event["RoundNumber"],
                    "driver_id": driver_id,
                    "team_name": row.get("TeamName", ""),
                    "position": row.get("Position", None),
                    "grid_position": row.get("GridPosition", None),
                    "status": row.get("Status", ""),
                    "points": row.get("Points", 0),
                    "fastest_lap_time": (
                        row.get("FastestLapTime").total_seconds()
                        if pd.notna(row.get("FastestLapTime"))
                        else None
                    ),
                })

        except Exception as e:
            print(f"  Error extracting results: {e}")

        return results

    def collect_season_data(self, year: int) -> Dict[str, List]:
        """Collect all FastF1 data for a season."""
        print(f"Collecting FastF1 data for {year}...")

        all_stints = []
        all_weather = []
        all_results = []

        schedule = self.get_event_schedule(year)
        if schedule.empty:
            print(f"  No schedule found for {year}")
            return {"stints": [], "weather": [], "results": []}

        # Filter to completed races only
        races = schedule[schedule["EventFormat"] != "testing"]

        for _, event in races.iterrows():
            round_num = event.get("RoundNumber", 0)
            event_name = event.get("EventName", f"Round {round_num}")

            if round_num == 0:
                continue

            print(f"  Loading {event_name} (Round {round_num})...")

            session = self.get_session_safely(year, round_num, "R")
            if session is None:
                continue

            # Extract data
            stints = self.extract_tyre_stints(session)
            weather = self.extract_weather(session)
            results = self.extract_driver_results(session)

            all_stints.extend(stints)
            if weather:
                all_weather.append(weather)
            all_results.extend(results)

        return {
            "stints": all_stints,
            "weather": all_weather,
            "results": all_results,
        }

    def collect_all_data(self) -> Dict[str, pd.DataFrame]:
        """Collect FastF1 data for all configured years."""
        print("=" * 60)
        print("FASTF1 DATA COLLECTOR")
        print(f"Year range: {self.start_year} - {self.end_year}")
        print("=" * 60)

        all_stints = []
        all_weather = []
        all_results = []

        for year in range(self.start_year, self.end_year + 1):
            try:
                data = self.collect_season_data(year)
                all_stints.extend(data["stints"])
                all_weather.extend(data["weather"])
                all_results.extend(data["results"])
            except Exception as e:
                print(f"Error collecting {year}: {e}")
                continue

        # Convert to DataFrames and save
        stints_df = pd.DataFrame(all_stints)
        weather_df = pd.DataFrame(all_weather)
        results_df = pd.DataFrame(all_results)

        # Save to CSV
        if not stints_df.empty:
            stints_df.to_csv(self.output_dir / "tyre_stints.csv", index=False)
            print(f"Saved {len(stints_df)} tyre stints")

        if not weather_df.empty:
            weather_df.to_csv(self.output_dir / "weather.csv", index=False)
            print(f"Saved {len(weather_df)} weather records")

        if not results_df.empty:
            results_df.to_csv(self.output_dir / "fastf1_results.csv", index=False)
            print(f"Saved {len(results_df)} FastF1 result records")

        print("=" * 60)
        print("FASTF1 DATA COLLECTION COMPLETE")
        print("=" * 60)

        return {
            "tyre_stints": stints_df,
            "weather": weather_df,
            "fastf1_results": results_df,
        }


def main():
    """Run FastF1 data collection."""
    import argparse

    parser = argparse.ArgumentParser(description="Collect F1 data from FastF1")
    parser.add_argument(
        "--start-year",
        type=int,
        default=FASTF1_START_YEAR,
        help=f"Start year (default: {FASTF1_START_YEAR})",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=FASTF1_END_YEAR,
        help=f"End year (default: {FASTF1_END_YEAR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=RAW_DATASET_DIR,
        help=f"Output directory (default: {RAW_DATASET_DIR})",
    )

    args = parser.parse_args()

    collector = FastF1Collector(
        start_year=args.start_year,
        end_year=args.end_year,
        output_dir=args.output_dir,
    )
    collector.collect_all_data()


if __name__ == "__main__":
    main()
