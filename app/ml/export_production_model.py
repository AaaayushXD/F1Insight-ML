"""
Export a single production pickle bundling all artifacts needed for inference.

Usage:
    python -m app.ml.export_production_model
"""

from pathlib import Path
from datetime import datetime
import hashlib
import json
import sys
import numpy as np
import joblib

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def export_production_model(output_dir: Path = None):
    """Bundle preprocessor, base models, meta-learner, and classifiers into one pickle."""
    output_dir = output_dir or OUTPUT_DIR
    ensemble_path = output_dir / "ensemble.joblib"
    preprocessor_path = output_dir / "preprocessor.joblib"
    report_path = output_dir / "evaluation_report.json"

    if not ensemble_path.exists():
        print(f"ERROR: {ensemble_path} not found. Run training first: python -m app.ml.train")
        sys.exit(1)
    if not preprocessor_path.exists():
        print(f"ERROR: {preprocessor_path} not found. Run training first.")
        sys.exit(1)

    # Load artifacts
    ensemble = joblib.load(ensemble_path)
    preprocessor = joblib.load(preprocessor_path)

    report = {}
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)

    # Load best podium classifier
    podium_model = None
    for name in ["xgboost", "gradient_boosting", "random_forest", "logistic"]:
        path = output_dir / f"model_classification_podium_{name}.joblib"
        if path.exists():
            podium_model = joblib.load(path)
            break

    # Build production artifact
    artifact = {
        "version": "2.0.0",
        "exported_at": datetime.now().isoformat(),
        "scaler": preprocessor["scaler"],
        "encoders": preprocessor["encoders"],
        "numeric_cols": preprocessor["numeric_cols"],
        "cat_cols": preprocessor["cat_cols"],
        "feature_names": preprocessor.get("feature_names", preprocessor["numeric_cols"] + preprocessor["cat_cols"]),
        "base_models": ensemble["base_models"],
        "meta_model": ensemble["meta_model"],
        "ensemble_scaler": ensemble["scaler"],
        "ensemble_encoders": ensemble["encoders"],
    }

    if podium_model is not None:
        artifact["podium_model"] = podium_model

    # Add training metadata
    best = report.get("best_model", {})
    artifact["metadata"] = {
        "best_regression": best.get("regression", "ensemble"),
        "test_mae": best.get("regression_mae"),
        "test_spearman": best.get("regression_spearman"),
        "train_val_gap": best.get("train_val_gap"),
        "temporal_split": report.get("temporal_split", {}),
        "data_summary": report.get("data_summary", {}),
        "success_criteria": report.get("success_criteria", {}),
        "hyperparameters": report.get("best_hyperparameters", {}),
    }

    # Save production pickle
    prod_path = output_dir / "f1insight_production_model.joblib"
    joblib.dump(artifact, prod_path)

    size_mb = prod_path.stat().st_size / (1024 * 1024)
    print(f"Production model exported: {prod_path}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Version: {artifact['version']}")
    print(f"  Base models: {list(artifact['base_models'].keys())}")
    print(f"  Has podium classifier: {podium_model is not None}")
    print(f"  Test MAE: {artifact['metadata'].get('test_mae', 'N/A')}")
    print(f"  Test Spearman: {artifact['metadata'].get('test_spearman', 'N/A')}")

    return prod_path


if __name__ == "__main__":
    export_production_model()
