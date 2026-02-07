"""
F1Insight configuration. Prefer environment variables for deployment.
"""
import os

# Ergast-compatible API (e.g. Jolpi). Set API_BASE_URL in .env to override.
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.jolpi.ca/ergast/f1").rstrip("/")

# Data paths (relative to project root)
RAW_DATASET_DIR = os.getenv("RAW_DATASET_DIR", "app/data/raw_dataset")
CLEAN_DATASET_DIR = os.getenv("CLEAN_DATASET_DIR", "app/data/cleaned_dataset")

# ML temporal split
TRAIN_SEASONS = (2014, 2021)
VAL_SEASONS = (2022, 2023)
TEST_SEASONS = (2024, 2025)
