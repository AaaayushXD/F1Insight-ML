import requests
import pandas as pd
import json
import time
import os

class F1DataFetcher:
    """Fetch all F1 data from Ergast API and save raw responses to CSV files."""
    
    BASE_URL = os.getenv("API_BASE_URL")
    if not BASE_URL:
        raise ValueError("API_BASE_URL is not set")
    
    def __init__(self, start_year: int = 2014, end_year: int = 2025):
        self.start_year = start_year
        self.end_year = end_year
        self.output_dir = "app/data/raw_dataset"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def fetch_data(self, endpoint: str, params: dict = None, max_retries: int = 5) -> dict:
        """Fetch data from API with error handling and retry logic for rate limiting."""
        url = f"{self.BASE_URL}/{endpoint}.json"
        if params:
            url += "?" + "&".join([f"{k}={v}" for k, v in params.items()])
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=30)
                
                # Handle rate limiting (429) with exponential backoff
                if response.status_code == 429:
                    wait_time = (2 ** attempt) + (attempt * 0.5)  # Exponential backoff: 1s, 2.5s, 5s, 8.5s, 13s
                    if attempt < max_retries - 1:
                        print(f"Rate limited on {endpoint}, waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Error fetching {endpoint}: 429 Too Many Requests (max retries exceeded)")
                        return None
                
                response.raise_for_status()
                time.sleep(0.5)  # Base rate limiting between successful requests
                return response.json()
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + (attempt * 0.5)
                    print(f"Error fetching {endpoint}: {e}, retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Error fetching {endpoint}: {e} (max retries exceeded)")
                    return None
        
        return None
    
    def flatten_dict(self, d, parent_key='', sep='_'):
        """Flatten nested dictionary structure."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                items.append((new_key, json.dumps(v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def get_seasons(self) -> pd.DataFrame:
        """Fetch all seasons data."""
        print("Fetching seasons...")
        all_data = []
        
        offset = 0
        limit = 100
        while True:
            data = self.fetch_data("seasons", params={"limit": limit, "offset": offset})
            if not data or 'MRData' not in data:
                break
            
            seasons = data['MRData']['SeasonTable'].get('Seasons', [])
            if not seasons:
                break
            
            for item in seasons:
                all_data.append(self.flatten_dict(item))
            
            offset += limit
            if len(seasons) < limit:
                break
        
        df = pd.DataFrame(all_data)
        df.to_csv(f"{self.output_dir}/seasons.csv", index=False)
        print(f"Saved {len(df)} seasons")
        return df
    
    def get_circuits(self) -> pd.DataFrame:
        """Fetch all circuits data."""
        print("Fetching circuits...")
        all_data = []
        
        offset = 0
        limit = 100
        while True:
            data = self.fetch_data("circuits", params={"limit": limit, "offset": offset})
            if not data or 'MRData' not in data:
                break
            
            circuits = data['MRData']['CircuitTable'].get('Circuits', [])
            if not circuits:
                break
            
            for item in circuits:
                all_data.append(self.flatten_dict(item))
            
            offset += limit
            if len(circuits) < limit:
                break
        
        df = pd.DataFrame(all_data)
        df.to_csv(f"{self.output_dir}/circuits.csv", index=False)
        print(f"Saved {len(df)} circuits")
        return df
    
    def get_races(self) -> pd.DataFrame:
        """Fetch race schedule data for all years."""
        print("Fetching races...")
        all_data = []
        
        for year in range(self.start_year, self.end_year + 1):
            data = self.fetch_data(f"{year}/races")
            
            if data and 'MRData' in data:
                races = data['MRData']['RaceTable'].get('Races', [])
                for item in races:
                    all_data.append(self.flatten_dict(item))
        
        df = pd.DataFrame(all_data)
        df.to_csv(f"{self.output_dir}/races.csv", index=False)
        print(f"Saved {len(df)} races")
        return df
    
    def get_constructors(self) -> pd.DataFrame:
        """Fetch all constructors data."""
        print("Fetching constructors...")
        all_data = []
        
        offset = 0
        limit = 100
        while True:
            data = self.fetch_data("constructors", params={"limit": limit, "offset": offset})
            if not data or 'MRData' not in data:
                break
            
            constructors = data['MRData']['ConstructorTable'].get('Constructors', [])
            if not constructors:
                break
            
            for item in constructors:
                all_data.append(self.flatten_dict(item))
            
            offset += limit
            if len(constructors) < limit:
                break
        
        df = pd.DataFrame(all_data)
        df.to_csv(f"{self.output_dir}/constructors.csv", index=False)
        print(f"Saved {len(df)} constructors")
        return df
    
    def get_drivers(self) -> pd.DataFrame:
        """Fetch all drivers data."""
        print("Fetching drivers...")
        all_data = []
        
        offset = 0
        limit = 100
        while True:
            data = self.fetch_data("drivers", params={"limit": limit, "offset": offset})
            if not data or 'MRData' not in data:
                break
            
            drivers = data['MRData']['DriverTable'].get('Drivers', [])
            if not drivers:
                break
            
            for item in drivers:
                all_data.append(self.flatten_dict(item))
            
            offset += limit
            if len(drivers) < limit:
                break
        
        df = pd.DataFrame(all_data)
        df.to_csv(f"{self.output_dir}/drivers.csv", index=False)
        print(f"Saved {len(df)} drivers")
        return df
    
    def get_results(self) -> pd.DataFrame:
        """Fetch race results for all years."""
        print("Fetching results...")
        all_data = []
        
        for year in range(self.start_year, self.end_year + 1):
            data = self.fetch_data(f"{year}/results")
            
            if data and 'MRData' in data:
                races = data['MRData']['RaceTable'].get('Races', [])
                for race in races:
                    race_info = {k: v for k, v in race.items() if k != 'Results'}
                    flattened_race = self.flatten_dict(race_info)
                    
                    for result in race.get('Results', []):
                        flattened_result = self.flatten_dict(result)
                        combined = {**flattened_race, **flattened_result}
                        all_data.append(combined)
        
        df = pd.DataFrame(all_data)
        df.to_csv(f"{self.output_dir}/results.csv", index=False)
        print(f"Saved {len(df)} results")
        return df
    
    def get_sprint(self) -> pd.DataFrame:
        """Fetch sprint race results for all years."""
        print("Fetching sprint results...")
        all_data = []
        
        for year in range(self.start_year, self.end_year + 1):
            data = self.fetch_data(f"{year}/sprint")
            
            if data and 'MRData' in data:
                races = data['MRData']['RaceTable'].get('Races', [])
                for race in races:
                    race_info = {k: v for k, v in race.items() if k != 'SprintResults'}
                    flattened_race = self.flatten_dict(race_info)
                    
                    for result in race.get('SprintResults', []):
                        flattened_result = self.flatten_dict(result)
                        combined = {**flattened_race, **flattened_result}
                        all_data.append(combined)
        
        df = pd.DataFrame(all_data)
        if len(df) > 0:
            df.to_csv(f"{self.output_dir}/sprint.csv", index=False)
            print(f"Saved {len(df)} sprint results")
        else:
            print("No sprint data available")
        return df
    
    def get_qualifying(self) -> pd.DataFrame:
        """Fetch qualifying results for all years."""
        print("Fetching qualifying...")
        all_data = []
        
        for year in range(self.start_year, self.end_year + 1):
            data = self.fetch_data(f"{year}/qualifying")
            
            if data and 'MRData' in data:
                races = data['MRData']['RaceTable'].get('Races', [])
                for race in races:
                    race_info = {k: v for k, v in race.items() if k != 'QualifyingResults'}
                    flattened_race = self.flatten_dict(race_info)
                    
                    for qual in race.get('QualifyingResults', []):
                        flattened_qual = self.flatten_dict(qual)
                        combined = {**flattened_race, **flattened_qual}
                        all_data.append(combined)
        
        df = pd.DataFrame(all_data)
        df.to_csv(f"{self.output_dir}/qualifying.csv", index=False)
        print(f"Saved {len(df)} qualifying results")
        return df
    
    def get_pitstops(self) -> pd.DataFrame:
        """Fetch pit stop data for all years."""
        print("Fetching pit stops...")
        all_data = []
        
        for year in range(self.start_year, self.end_year + 1):
            races_data = self.fetch_data(f"{year}/races")
            if not races_data or 'MRData' not in races_data:
                continue
            
            num_rounds = len(races_data['MRData']['RaceTable'].get('Races', []))
            
            for round_num in range(1, num_rounds + 1):
                data = self.fetch_data(f"{year}/{round_num}/pitstops")
                
                if data and 'MRData' in data:
                    races = data['MRData']['RaceTable'].get('Races', [])
                    for race in races:
                        race_info = {k: v for k, v in race.items() if k != 'PitStops'}
                        flattened_race = self.flatten_dict(race_info)
                        
                        for pitstop in race.get('PitStops', []):
                            flattened_pitstop = self.flatten_dict(pitstop)
                            combined = {**flattened_race, **flattened_pitstop}
                            all_data.append(combined)
        
        df = pd.DataFrame(all_data)
        df.to_csv(f"{self.output_dir}/pitstops.csv", index=False)
        print(f"Saved {len(df)} pit stops")
        return df
    
    def get_laps(self, limit_rounds: int = None) -> pd.DataFrame:
        """Fetch lap times. Warning: This generates LARGE datasets!
        Args:
            limit_rounds: Limit to first N rounds per year (None = all rounds)
        """
        print("Fetching lap times (this may take a while)...")
        all_data = []
        
        for year in range(self.start_year, self.end_year + 1):
            races_data = self.fetch_data(f"{year}/races")
            if not races_data or 'MRData' not in races_data:
                continue
            
            num_rounds = len(races_data['MRData']['RaceTable'].get('Races', []))
            if limit_rounds:
                num_rounds = min(num_rounds, limit_rounds)
            
            for round_num in range(1, num_rounds + 1):
                print(f"  {year} Round {round_num}/{num_rounds}...")
                data = self.fetch_data(f"{year}/{round_num}/laps")
                
                if data and 'MRData' in data:
                    races = data['MRData']['RaceTable'].get('Races', [])
                    for race in races:
                        race_info = {k: v for k, v in race.items() if k != 'Laps'}
                        flattened_race = self.flatten_dict(race_info)
                        
                        for lap in race.get('Laps', []):
                            lap_info = {k: v for k, v in lap.items() if k != 'Timings'}
                            
                            for timing in lap.get('Timings', []):
                                flattened_timing = self.flatten_dict(timing)
                                combined = {**flattened_race, **lap_info, **flattened_timing}
                                all_data.append(combined)
        
        df = pd.DataFrame(all_data)
        df.to_csv(f"{self.output_dir}/laps.csv", index=False)
        print(f"Saved {len(df)} lap times")
        return df
    
    def get_driver_standings(self) -> pd.DataFrame:
        """Fetch driver championship standings for all years."""
        print("Fetching driver standings...")
        all_data = []
        
        for year in range(self.start_year, self.end_year + 1):
            data = self.fetch_data(f"{year}/driverStandings")
            
            if data and 'MRData' in data:
                standings_lists = data['MRData']['StandingsTable'].get('StandingsLists', [])
                for standings_list in standings_lists:
                    standings_info = {k: v for k, v in standings_list.items() if k != 'DriverStandings'}
                    
                    for standing in standings_list.get('DriverStandings', []):
                        flattened_standing = self.flatten_dict(standing)
                        combined = {**standings_info, **flattened_standing}
                        all_data.append(combined)
        
        df = pd.DataFrame(all_data)
        df.to_csv(f"{self.output_dir}/driver_standings.csv", index=False)
        print(f"Saved {len(df)} driver standings")
        return df
    
    def get_constructor_standings(self) -> pd.DataFrame:
        """Fetch constructor championship standings for all years."""
        print("Fetching constructor standings...")
        all_data = []
        
        for year in range(self.start_year, self.end_year + 1):
            data = self.fetch_data(f"{year}/constructorStandings")
            
            if data and 'MRData' in data:
                standings_lists = data['MRData']['StandingsTable'].get('StandingsLists', [])
                for standings_list in standings_lists:
                    standings_info = {k: v for k, v in standings_list.items() if k != 'ConstructorStandings'}
                    
                    for standing in standings_list.get('ConstructorStandings', []):
                        flattened_standing = self.flatten_dict(standing)
                        combined = {**standings_info, **flattened_standing}
                        all_data.append(combined)
        
        df = pd.DataFrame(all_data)
        df.to_csv(f"{self.output_dir}/constructor_standings.csv", index=False)
        print(f"Saved {len(df)} constructor standings")
        return df
    
    def get_status(self) -> pd.DataFrame:
        """Fetch all status codes (finishing statuses)."""
        print("Fetching status codes...")
        all_data = []
        
        offset = 0
        limit = 100
        while True:
            data = self.fetch_data("status", params={"limit": limit, "offset": offset})
            if not data or 'MRData' not in data:
                break
            
            statuses = data['MRData']['StatusTable'].get('Status', [])
            if not statuses:
                break
            
            for item in statuses:
                all_data.append(self.flatten_dict(item))
            
            offset += limit
            if len(statuses) < limit:
                break
        
        df = pd.DataFrame(all_data)
        df.to_csv(f"{self.output_dir}/status.csv", index=False)
        print(f"Saved {len(df)} status codes")
        return df
    
    def fetch_all_data(self, include_laps: bool = False):
        """
        Fetch ALL available F1 data and save to CSV files.
        
        Args:
            include_laps: Include lap-by-lap timing data (WARNING: creates very large files!)
        """
        print("="*60)
        print("F1 DATA FETCHER - FETCHING ALL AVAILABLE DATA")
        print(f"Year range: {self.start_year} - {self.end_year}")
        print("="*60)
        print()
        
        # Master/lookup tables (not year-specific)
        self.get_seasons()
        self.get_circuits()
        self.get_constructors()
        self.get_drivers()
        self.get_status()
        
        print()
        print("-"*60)
        print("YEAR-SPECIFIC DATA")
        print("-"*60)
        print()
        
        # Year-specific data
        self.get_races()
        self.get_results()
        self.get_sprint()
        self.get_qualifying()
        self.get_pitstops()
        self.get_driver_standings()
        self.get_constructor_standings()
        
        if include_laps:
            print()
            print("⚠️  WARNING: Fetching lap times - this will take significant time!")
            print("-"*60)
            self.get_laps()
        
        print()
        print("="*60)
        print("DATA COLLECTION COMPLETE!")
        print(f"All CSV files saved to: {self.output_dir}/")
        print("="*60)
        print()
        print("Files created:")
        print("  Master/Lookup Tables:")
        print("    - seasons.csv")
        print("    - circuits.csv")
        print("    - constructors.csv")
        print("    - drivers.csv")
        print("    - status.csv")
        print()
        print("  Race Data:")
        print("    - races.csv")
        print("    - results.csv")
        print("    - sprint.csv")
        print("    - qualifying.csv")
        print("    - pitstops.csv")
        if include_laps:
            print("    - laps.csv")
        print()
        print("  Standings:")
        print("    - driver_standings.csv")
        print("    - constructor_standings.csv")


