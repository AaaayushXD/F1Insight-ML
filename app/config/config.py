"""
F1Insight configuration. Prefer environment variables for deployment.
"""
import os
from pathlib import Path

# Ergast-compatible API (e.g. Jolpi). Set API_BASE_URL in .env to override.
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.jolpi.ca/ergast/f1").rstrip("/")

# Data paths (relative to project root)
RAW_DATASET_DIR = os.getenv("RAW_DATASET_DIR", "app/data/raw_dataset")
CLEAN_DATASET_DIR = os.getenv("CLEAN_DATASET_DIR", "app/data/cleaned_dataset")
PROCESSED_DATASET_DIR = os.getenv("PROCESSED_DATASET_DIR", "app/data/processed_dataset")

# FastF1 cache directory for session data
FASTF1_CACHE_DIR = os.getenv("FASTF1_CACHE_DIR", "app/data/fastf1_cache")

# ML temporal split
TRAIN_SEASONS = (2014, 2021)
VAL_SEASONS = (2022, 2023)
TEST_SEASONS = (2024, 2025)

# FastF1 data collection settings
FASTF1_START_YEAR = 2018  # FastF1 has reliable data from 2018 onwards
FASTF1_END_YEAR = 2025    

# Feature flags for optional data sources
USE_FASTF1_DATA = os.getenv("USE_FASTF1_DATA", "true").lower() == "true"
USE_WEATHER_DATA = os.getenv("USE_WEATHER_DATA", "true").lower() == "true"
USE_TYRE_DATA = os.getenv("USE_TYRE_DATA", "true").lower() == "true"

# Random state for reproducibility
RANDOM_STATE = 42
