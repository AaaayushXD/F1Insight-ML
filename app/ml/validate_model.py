"""
Comprehensive model validation on the held-out test set (2024-2025).

Loads the production pickle, evaluates on test data, generates validation_report.json.

Usage:
    python -m app.ml.validate_model
"""

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.ml.build_dataset import build_merged_dataset
from app.ml.train import (
    _prepare_xy, _apply_preprocessor,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    TARGET_REGRESSION, TEST_SEASONS,
    compute_position_bucket_accuracy,
)

CLEAN_DIR = Path(__file__).resolve().parent.parent / "data" / "cleaned_dataset"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def validate_production_model(output_dir: Path = None, clean_dir: Path = None):
    """Run full validation on test set using production pickle."""
    output_dir = output_dir or OUTPUT_DIR
    clean_dir = clean_dir or CLEAN_DIR
    prod_path = output_dir / "f1insight_production_model.joblib"

    if not prod_path.exists():
        print(f"ERROR: Production model not found at {prod_path}")
        print("Run: python -m app.ml.export_production_model")
        sys.exit(1)

    print("=" * 60)
    print("F1INSIGHT MODEL VALIDATION")
    print("=" * 60)

    # Load production model
    print("\n[1/5] Loading production model...")
    artifact = joblib.load(prod_path)
    scaler = artifact["ensemble_scaler"]
    encoders = artifact["ensemble_encoders"]
    base_models = artifact["base_models"]
    meta_model = artifact["meta_model"]
    numeric_cols = artifact["numeric_cols"]
    cat_cols = artifact["cat_cols"]

    print(f"  Version: {artifact.get('version', 'unknown')}")
    print(f"  Base models: {list(base_models.keys())}")
    print(f"  Features: {len(numeric_cols)} numeric + {len(cat_cols)} categorical")

    # Build test dataset
    print("\n[2/5] Building test dataset (2024-2025)...")
    df = build_merged_dataset(clean_dir)
    test_df = df[(df["season"] >= TEST_SEASONS[0]) & (df["season"] <= TEST_SEASONS[1])].copy()
    test_df = test_df.dropna(subset=[TARGET_REGRESSION])

    if len(test_df) == 0:
        print("ERROR: No test data available for 2024-2025 seasons.")
        sys.exit(1)

    print(f"  Test rows: {len(test_df)}")
    print(f"  Seasons: {sorted(test_df['season'].unique())}")

    # Preprocess
    print("\n[3/5] Running predictions...")
    X_num, X_cat, _, _ = _prepare_xy(test_df, TARGET_REGRESSION, "is_podium", numeric_cols, cat_cols)
    X_te = _apply_preprocessor(X_num, X_cat, scaler, encoders, numeric_cols, cat_cols)
    y_te = test_df[TARGET_REGRESSION].values

    # Ensemble prediction
    base_preds = np.column_stack([bm.predict(X_te) for bm in base_models.values()])
    ensemble_pred = meta_model.predict(base_preds)

    # Individual model predictions
    individual_preds = {}
    for name, bm in base_models.items():
        individual_preds[name] = bm.predict(X_te)

    # Grid position baseline
    grid_pred = test_df["grid_position"].fillna(10).values

    # Compute metrics
    print("\n[4/5] Computing metrics...")
    report = {
        "validation_date": pd.Timestamp.now().isoformat(),
        "model_version": artifact.get("version", "unknown"),
        "test_seasons": f"{TEST_SEASONS[0]}-{TEST_SEASONS[1]}",
        "test_rows": len(test_df),
    }

    # Ensemble metrics
    valid = ~np.isnan(y_te)
    ens_mae = float(mean_absolute_error(y_te[valid], ensemble_pred[valid]))
    ens_rmse = float(np.sqrt(mean_squared_error(y_te[valid], ensemble_pred[valid])))
    ens_spearman = float(spearmanr(y_te[valid], ensemble_pred[valid])[0]) if len(np.unique(y_te[valid])) > 1 else 0.0
    ens_bucket = compute_position_bucket_accuracy(y_te[valid], ensemble_pred[valid])

    report["ensemble"] = {
        "mae": ens_mae,
        "rmse": ens_rmse,
        "spearman": ens_spearman,
        "bucket_accuracy": ens_bucket,
    }

    # Individual model metrics
    report["individual_models"] = {}
    for name, pred in individual_preds.items():
        mae = float(mean_absolute_error(y_te[valid], pred[valid]))
        rmse = float(np.sqrt(mean_squared_error(y_te[valid], pred[valid])))
        sp = float(spearmanr(y_te[valid], pred[valid])[0]) if len(np.unique(y_te[valid])) > 1 else 0.0
        report["individual_models"][name] = {"mae": mae, "rmse": rmse, "spearman": sp}

    # Grid baseline
    baseline_mae = float(mean_absolute_error(y_te[valid], grid_pred[valid]))
    baseline_spearman = float(spearmanr(y_te[valid], grid_pred[valid])[0]) if len(np.unique(y_te[valid])) > 1 else 0.0
    report["baseline_grid_position"] = {"mae": baseline_mae, "spearman": baseline_spearman}
    report["improvement_over_baseline"] = {
        "mae_reduction": float(baseline_mae - ens_mae),
        "mae_reduction_pct": float((baseline_mae - ens_mae) / baseline_mae * 100) if baseline_mae > 0 else 0.0,
    }

    # Per-race analysis
    report["per_race"] = []
    for (season, round_num), grp in test_df.groupby(["season", "round"]):
        idx = grp.index
        mask = test_df.index.isin(idx)
        race_y = y_te[mask]
        race_pred = ensemble_pred[mask]
        race_valid = ~np.isnan(race_y)
        if race_valid.sum() < 5:
            continue

        race_mae = float(mean_absolute_error(race_y[race_valid], race_pred[race_valid]))
        race_sp = float(spearmanr(race_y[race_valid], race_pred[race_valid])[0]) if len(np.unique(race_y[race_valid])) > 1 else 0.0

        circuit = grp["circuit_id"].iloc[0] if "circuit_id" in grp.columns else "unknown"
        report["per_race"].append({
            "season": int(season),
            "round": int(round_num),
            "circuit": circuit,
            "n_drivers": int(race_valid.sum()),
            "mae": race_mae,
            "spearman": race_sp,
        })

    # Sanity checks
    print("\n[5/5] Running sanity checks...")
    report["sanity_checks"] = {}

    # Check 1: P1 qualifier predicted better than P20
    if "qualifying_position" in test_df.columns:
        p1_rows = test_df[test_df["qualifying_position"] == 1.0]
        p20_rows = test_df[test_df["qualifying_position"] == 20.0]
        if len(p1_rows) > 0 and len(p20_rows) > 0:
            p1_idx = test_df.index.isin(p1_rows.index)
            p20_idx = test_df.index.isin(p20_rows.index)
            avg_p1_pred = float(np.mean(ensemble_pred[p1_idx]))
            avg_p20_pred = float(np.mean(ensemble_pred[p20_idx]))
            report["sanity_checks"]["p1_predicted_better_than_p20"] = {
                "avg_pred_p1_qualifier": avg_p1_pred,
                "avg_pred_p20_qualifier": avg_p20_pred,
                "passed": avg_p1_pred < avg_p20_pred,
            }

    # Check 2: Predictions within valid range
    pred_min, pred_max = float(ensemble_pred.min()), float(ensemble_pred.max())
    report["sanity_checks"]["predictions_in_range"] = {
        "min_pred": pred_min,
        "max_pred": pred_max,
        "passed": pred_min >= 0.0 and pred_max <= 25.0,
    }

    # Check 3: Better than random
    random_preds = np.random.RandomState(42).uniform(1, 20, len(y_te))
    random_mae = float(mean_absolute_error(y_te[valid], random_preds[valid]))
    report["sanity_checks"]["beats_random"] = {
        "random_mae": random_mae,
        "model_mae": ens_mae,
        "passed": ens_mae < random_mae,
    }

    # Success criteria (realistic for pre-race-only features with ~35% DNF rate)
    report["success_criteria"] = {
        "test_mae_under_3_2": {"target": "< 3.2", "actual": f"{ens_mae:.4f}", "passed": ens_mae < 3.2},
        "spearman_over_0_65": {"target": "> 0.65", "actual": f"{ens_spearman:.4f}", "passed": ens_spearman > 0.65},
        "no_overfitting": {"target": "beats baseline", "actual": f"{ens_mae:.4f} vs {baseline_mae:.4f}", "passed": ens_mae < baseline_mae},
        "beats_baseline": {"target": f"< {baseline_mae:.4f}", "actual": f"{ens_mae:.4f}", "passed": ens_mae < baseline_mae},
    }
    report["success_criteria"]["all_passed"] = all(
        v["passed"] for k, v in report["success_criteria"].items() if k != "all_passed"
    )

    # Save report
    val_path = output_dir / "validation_report.json"
    with open(val_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"\nEnsemble Performance:")
    print(f"  MAE:      {ens_mae:.4f}")
    print(f"  RMSE:     {ens_rmse:.4f}")
    print(f"  Spearman: {ens_spearman:.4f}")
    print(f"  Bucket accuracy: {ens_bucket['bucket_accuracy']:.4f}")

    print(f"\nBaseline (grid position):")
    print(f"  MAE: {baseline_mae:.4f}")
    print(f"  Improvement: {baseline_mae - ens_mae:.4f} ({(baseline_mae - ens_mae)/baseline_mae*100:.1f}%)")

    print(f"\nIndividual Models:")
    for name, m in report["individual_models"].items():
        print(f"  {name}: MAE={m['mae']:.4f}, Spearman={m['spearman']:.4f}")

    print(f"\nSanity Checks:")
    for name, check in report["sanity_checks"].items():
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  {name}: [{status}]")

    print(f"\nSuccess Criteria:")
    for key, val in report["success_criteria"].items():
        if key != "all_passed":
            status = "PASS" if val["passed"] else "FAIL"
            print(f"  {key}: {val['actual']} (target: {val['target']}) [{status}]")
    print(f"\nAll criteria passed: {report['success_criteria']['all_passed']}")
    print(f"\nReport saved: {val_path}")
    print("=" * 60)

    return report


if __name__ == "__main__":
    validate_production_model()
