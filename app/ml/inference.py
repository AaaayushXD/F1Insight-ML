"""
F1Insight inference: load trained models and preprocessor, run prediction.

Supports the enhanced feature set including weather, tyre, and form features.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# Pre-race features only (must match training — no race-time data)
NUMERIC_COLS = [
    # Core positioning features
    "qualifying_position",
    "grid_position",
    # Driver standings
    "driver_prior_points",
    "driver_prior_wins",
    "driver_prior_position",
    # Constructor standings
    "constructor_prior_points",
    "constructor_prior_wins",
    "constructor_prior_position",
    # Historical pitstop features (pre-race knowable)
    "historical_avg_stops_at_circuit",
    "driver_historical_avg_stops",
    # Circuit features
    "circuit_lat",
    "circuit_lng",
    "circuit_avg_positions_gained",
    # Rolling performance features
    "driver_recent_avg_finish",
    "driver_circuit_avg_finish",
    "driver_avg_positions_gained",
    "constructor_recent_avg_finish",
    # Form and head-to-head
    "driver_form_trend",
    "gap_to_teammate_quali",
    # Pre-race derived features
    "season_round_number",
    "constructor_relative_performance",
    # Reliability features
    "driver_dnf_rate",
    "constructor_dnf_rate",
    # Weather features
    "track_temp",
    "air_temp",
    "humidity",
    "is_wet_race",
    "wind_speed",
]
CAT_COLS = ["driver_id", "constructor_id", "circuit_id"]


def _load_artifacts(output_dir: Path):
    # Prefer production pickle if available
    prod_path = output_dir / "f1insight_production_model.joblib"
    if prod_path.exists():
        artifact = joblib.load(prod_path)
        scaler = artifact["scaler"]
        encoders = artifact["encoders"]
        numeric_cols = artifact.get("numeric_cols", NUMERIC_COLS)
        cat_cols = artifact.get("cat_cols", CAT_COLS)
        # Ensemble: use meta_model + base_models
        if "meta_model" in artifact and "base_models" in artifact:
            return (scaler, encoders, numeric_cols, cat_cols), artifact, output_dir
        # Single model fallback
        if "regression_model" in artifact:
            return (scaler, encoders, numeric_cols, cat_cols), artifact["regression_model"], output_dir

    # Fallback to individual files
    path = output_dir / "preprocessor.joblib"
    if not path.exists():
        return None, None, None
    prep = joblib.load(path)
    scaler = prep.get("scaler")
    encoders = prep.get("encoders")
    numeric_cols = prep.get("numeric_cols", NUMERIC_COLS)
    cat_cols = prep.get("cat_cols", CAT_COLS)
    reg_path = output_dir / "model_regression_xgboost.joblib"
    if not reg_path.exists():
        reg_path = output_dir / "model_regression_random_forest.joblib"
    if not reg_path.exists():
        reg_path = output_dir / "model_regression_ridge.joblib"
    reg_model = joblib.load(reg_path) if reg_path.exists() else None
    return (scaler, encoders, numeric_cols, cat_cols), reg_model, output_dir


def predict(
    row: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Predict finish position and optionally podium/top10 probability.
    row: one row (or DataFrame with one row) with columns matching training.
    """
    from app.ml.train import _apply_preprocessor
    out_dir = output_dir or OUTPUT_DIR
    prep, reg_model, _ = _load_artifacts(out_dir)
    if prep is None or reg_model is None:
        return {"error": "models not found", "message": "Run training first (python -m app.ml.train)"}
    scaler, encoders, numeric_cols, cat_cols = prep
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    X_num_df = pd.DataFrame([{c: row.get(c, 0) if c in row.index else 0 for c in numeric_cols}]).reindex(columns=numeric_cols).fillna(0)
    X_cat_df = pd.DataFrame([{c: str(row.get(c, "__missing__")) for c in cat_cols}]).reindex(columns=cat_cols).fillna("__missing__")
    X = _apply_preprocessor(X_num_df, X_cat_df, scaler, encoders, numeric_cols, cat_cols)

    # Handle ensemble model (production pickle with meta_model + base_models)
    if isinstance(reg_model, dict) and "meta_model" in reg_model and "base_models" in reg_model:
        base_preds = []
        for name, bm in reg_model["base_models"].items():
            base_preds.append(bm.predict(X)[0])
        meta_X = np.array([base_preds])
        pred_position = float(reg_model["meta_model"].predict(meta_X)[0])
    else:
        pred_position = float(reg_model.predict(X)[0])

    result = {"predicted_finish_position": max(1.0, min(20.0, round(pred_position, 1)))}

    # Podium probability
    if isinstance(reg_model, dict) and "podium_model" in reg_model:
        clf = reg_model["podium_model"]
        if hasattr(clf, "predict_proba"):
            result["podium_probability"] = float(clf.predict_proba(X)[0, 1])
    else:
        podium_path = out_dir / "model_classification_podium_xgboost.joblib"
        if not podium_path.exists():
            podium_path = out_dir / "model_classification_podium_random_forest.joblib"
        if podium_path.exists():
            clf = joblib.load(podium_path)
            if hasattr(clf, "predict_proba"):
                result["podium_probability"] = float(clf.predict_proba(X)[0, 1])
    return result
