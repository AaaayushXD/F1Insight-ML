"""
F1Insight full pipeline: optional collect -> clean -> train.
Use when results.csv is missing or after updating data range.
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "app" / "data" / "raw_dataset"
CLEAN_DIR = PROJECT_ROOT / "app" / "data" / "cleaned_dataset"


def run_collect(start_year: int = 2014, end_year: int = 2025, include_laps: bool = False) -> None:
    """Run data collection (requires API_BASE_URL in env or config default)."""
    from app.config.config import API_BASE_URL
    from app.scripts.collect_data import F1DataFetcher
    fetcher = F1DataFetcher(start_year=start_year, end_year=end_year, base_url=API_BASE_URL)
    fetcher.fetch_all_data(include_laps=include_laps)


def run_clean() -> None:
    """Run cleaning and write to cleaned_dataset."""
    from app.scripts.clean_data import run_cleaning
    run_cleaning(raw_dir=str(RAW_DIR), clean_dir=str(CLEAN_DIR), filter_races_season_range=(2014, 2025))


def run_train() -> None:
    """Run ML training and write models + report to app/ml/outputs."""
    from app.ml.train import run_training
    run_training(clean_dir=CLEAN_DIR, output_dir=PROJECT_ROOT / "app" / "ml" / "outputs")


def main():
    parser = argparse.ArgumentParser(description="F1Insight pipeline: collect -> clean -> train")
    parser.add_argument("--skip-collect", action="store_true", help="Skip data collection (use existing raw CSVs)")
    parser.add_argument("--collect-only", action="store_true", help="Only run collection, then exit")
    parser.add_argument("--clean-only", action="store_true", help="Only run cleaning, then exit")
    parser.add_argument("--train-only", action="store_true", help="Only run training (requires cleaned data)")
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--include-laps", action="store_true", help="Include lap data (large)")
    args = parser.parse_args()

    if args.train_only:
        run_train()
        return
    if args.clean_only:
        run_clean()
        return
    if args.collect_only:
        run_collect(start_year=args.start_year, end_year=args.end_year, include_laps=args.include_laps)
        return

    if not args.skip_collect:
        need = not (RAW_DIR / "results.csv").exists()
        if need:
            print("results.csv not found in raw_dataset; running collection...")
            run_collect(start_year=args.start_year, end_year=args.end_year, include_laps=args.include_laps)
        else:
            print("results.csv found; skipping collection (use without --skip-collect to re-run).")

    print("Running cleaning...")
    run_clean()
    print("Running training...")
    run_train()
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
