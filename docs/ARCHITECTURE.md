# F1Insight System Architecture

## Overview

F1Insight is a Formula 1 race prediction and pit-stop strategy recommendation system that combines machine learning models with Monte Carlo simulation. The system is built as a FastAPI backend providing RESTful APIs for predictions and data access.

## System Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              F1Insight System                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                 │
│  │  Data Layer  │────▶│   ML Layer   │────▶│  API Layer   │                 │
│  └──────────────┘     └──────────────┘     └──────────────┘                 │
│         │                    │                    │                          │
│         ▼                    ▼                    ▼                          │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                 │
│  │ Ergast API   │     │   Training   │     │  FastAPI     │                 │
│  │ FastF1 API   │     │   Inference  │     │  /docs       │                 │
│  │ CSV Storage  │     │   Strategy   │     │  Endpoints   │                 │
│  └──────────────┘     └──────────────┘     └──────────────┘                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
F1Insight-ML/
├── app/
│   ├── config/
│   │   └── config.py           # Configuration settings
│   ├── data/
│   │   ├── raw_dataset/        # Raw CSV files from APIs
│   │   ├── cleaned_dataset/    # Normalized, cleaned CSVs
│   │   ├── processed_dataset/  # ML-ready merged data
│   │   └── fastf1_cache/       # FastF1 session cache
│   ├── ml/
│   │   ├── build_dataset.py    # Feature engineering
│   │   ├── train.py            # Model training pipeline
│   │   ├── inference.py        # Prediction service
│   │   ├── strategy.py         # Strategy recommendation
│   │   └── outputs/            # Trained models & reports
│   ├── scripts/
│   │   ├── collect_data.py     # Ergast API data collection
│   │   ├── collect_fastf1.py   # FastF1 data collection
│   │   └── clean_data.py       # Data cleaning pipeline
│   └── main.py                 # FastAPI application
├── tests/                      # Test suite
├── docs/                       # Documentation
└── requirements.txt            # Python dependencies
```

## Data Flow

### 1. Data Collection Pipeline

```
External APIs                    Raw Storage                 Cleaned Storage
┌───────────────┐               ┌─────────────┐             ┌─────────────────┐
│  Ergast API   │──────────────▶│ raw_dataset │────────────▶│ cleaned_dataset │
│  (Jolpi)      │   collect     │   12 CSVs   │   clean     │    12 CSVs      │
└───────────────┘   _data.py    └─────────────┘   _data.py  └─────────────────┘
                                                                    │
┌───────────────┐               ┌─────────────┐                     │
│   FastF1      │──────────────▶│ tyre_stints │─────────────────────┤
│   Library     │   collect     │ weather.csv │                     │
└───────────────┘   _fastf1.py  └─────────────┘                     │
                                                                    ▼
                                                            ┌─────────────────┐
                                                            │  ML Dataset     │
                                                            │  (merged)       │
                                                            └─────────────────┘
```

### 2. Data Entities

| Entity | Source | Description |
|--------|--------|-------------|
| seasons | Ergast | F1 seasons (2014-2025) |
| circuits | Ergast | Track information with coordinates |
| drivers | Ergast | Driver profiles and nationalities |
| constructors | Ergast | Team information |
| races | Ergast | Race schedule and metadata |
| results | Ergast | Race finish results |
| qualifying | Ergast | Qualifying session results |
| sprint | Ergast | Sprint race results |
| pitstops | Ergast | Pit stop timings |
| driver_standings | Ergast | Championship standings |
| constructor_standings | Ergast | Team standings |
| tyre_stints | FastF1 | Tyre compound and stint data |
| weather | FastF1 | Track and weather conditions |

### 3. ML Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  build_dataset  │────▶│     train       │────▶│   inference     │
│  (31 features)  │     │  (4 models)     │     │  (prediction)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
  Feature Types:          Model Types:            Outputs:
  - 27 numeric            - Ridge                 - Finish position
  - 4 categorical         - Random Forest         - Podium probability
                          - Gradient Boosting     - Top-10 probability
                          - XGBoost
```

## Component Details

### 1. Data Layer (`app/scripts/`)

#### collect_data.py
- **Purpose**: Fetch F1 data from Ergast-compatible API
- **Features**:
  - Rate limiting with exponential backoff
  - Retry logic for failed requests
  - Pagination handling
  - JSON flattening for nested responses

#### collect_fastf1.py
- **Purpose**: Collect enhanced data from FastF1 library
- **Features**:
  - Tyre compound extraction per stint
  - Weather data (track temp, air temp, humidity, rain)
  - Lap time analysis
  - Session caching for performance

#### clean_data.py
- **Purpose**: Normalize and standardize raw data
- **Features**:
  - Column name normalization (snake_case)
  - ID standardization (lowercase, trimmed)
  - Duration parsing (mm:ss.fff to seconds)
  - DNF status detection
  - Foreign key validation

### 2. ML Layer (`app/ml/`)

#### build_dataset.py
- **Purpose**: Create ML-ready feature matrix
- **Features**:
  - One row per (season, round, driver_id)
  - Prior-round standings (no future leakage)
  - Rolling performance averages
  - Circuit-specific history
  - Weather and tyre features
  - Form trend calculation

#### train.py
- **Purpose**: Train and evaluate ML models
- **Features**:
  - Temporal train/val/test split
  - Multiple model architectures
  - Feature importance analysis
  - Success criteria evaluation
  - Model artifact serialization

#### inference.py
- **Purpose**: Load models and make predictions
- **Features**:
  - Preprocessor loading
  - Feature transformation
  - Model selection (best available)
  - Probability estimation

#### strategy.py
- **Purpose**: Strategy recommendation via simulation
- **Features**:
  - KMeans clustering of pit patterns
  - Monte Carlo position simulation
  - Degradation/traffic uncertainty
  - Strategy ranking

### 3. API Layer (`app/main.py`)

#### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info and docs link |
| `/health` | GET | Health check |
| `/docs` | GET | Swagger UI documentation |
| `/redoc` | GET | ReDoc documentation |
| `/collect` | GET | Trigger data collection |
| `/predict` | GET | Single driver prediction |
| `/strategy` | GET | Strategy recommendation |
| `/api/seasons` | GET | List available seasons |
| `/api/races` | GET | List races for a season |
| `/api/drivers` | GET | List drivers |
| `/api/predictions/race` | GET | All predictions for a race |

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Runtime | Python | 3.9+ |
| Web Framework | FastAPI | 0.104.1 |
| ML Framework | scikit-learn | 1.3+ |
| Gradient Boosting | XGBoost | 2.0+ |
| Data Processing | pandas | 2.0+ |
| Numerical Computing | NumPy | 1.24+ |
| Model Persistence | joblib | 1.3+ |
| F1 Data | FastF1 | 3.4+ |
| HTTP Client | requests | 2.31+ |
| ASGI Server | uvicorn | 0.24+ |

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| API_BASE_URL | https://api.jolpi.ca/ergast/f1 | Ergast API endpoint |
| RAW_DATASET_DIR | app/data/raw_dataset | Raw data storage |
| CLEAN_DATASET_DIR | app/data/cleaned_dataset | Cleaned data storage |
| FASTF1_CACHE_DIR | app/data/fastf1_cache | FastF1 cache location |
| USE_FASTF1_DATA | true | Enable FastF1 features |

### ML Configuration

| Setting | Value | Description |
|---------|-------|-------------|
| TRAIN_SEASONS | 2014-2021 | Training data range |
| VAL_SEASONS | 2022-2023 | Validation data range |
| TEST_SEASONS | 2024-2025 | Test data range |
| RANDOM_STATE | 42 | Reproducibility seed |

## Deployment

### Local Development

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Data pipeline
python -m app.scripts.collect_data
python -m app.scripts.clean_data
python -m app.ml.train

# Run API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### API Access

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

## Security Considerations

1. **Rate Limiting**: External API calls use exponential backoff
2. **Input Validation**: Query parameters are validated with FastAPI
3. **CORS**: Configured for API access (adjustable in production)
4. **No Authentication**: Current version is open (add OAuth2 for production)

## Performance Considerations

1. **Caching**: FastF1 uses local cache for session data
2. **Model Loading**: Models loaded on first request, then cached
3. **Dataset Building**: Can be slow for full dataset; consider pre-computing
4. **Async**: FastAPI supports async endpoints for I/O-bound operations

## Future Enhancements

1. **Real-time Data**: WebSocket for live race updates
2. **Model Versioning**: MLflow integration for model tracking
3. **Feature Store**: Centralized feature management
4. **A/B Testing**: Model comparison in production
5. **Telemetry Integration**: Car telemetry for strategy refinement
