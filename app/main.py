from fastapi import FastAPI
from dotenv import load_dotenv
load_dotenv()
from app.scripts.collect_data import F1DataFetcher

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello, World!"}


@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/collect_f1_data")
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

    
    fetcher = F1DataFetcher(start_year=start_year, end_year=end_year)
    fetcher.fetch_all_data(include_laps=include_laps)
    
    return fetcher
