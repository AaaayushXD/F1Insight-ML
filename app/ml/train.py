"""
F1Insight ML training: race outcome prediction, podium/top-10/DNF probability.

Temporal split: train 2014-2021, val 2022-2023, test 2024-2025.
Models: Ridge, RandomForest, GradientBoosting, XGBoost (regression + classification).

Enhanced features:
- Weather data (track/air temp, wet race indicator)
- Tyre strategy (compounds, stint info)
- Driver form trends
- Team relative performance
"""

from pathlib import Path
import json
import sys
import warnings
import numpy as np
import pandas as pd

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge, LogisticRegression
except ImportError as e:
    print("Missing dependency: scikit-learn. Install with: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr
import joblib

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from app.ml.build_dataset import build_merged_dataset

CLEAN_DIR = Path(__file__).resolve().parent.parent / "data" / "cleaned_dataset"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
RANDOM_STATE = 42

# Temporal split (no shuffle across seasons)
TRAIN_SEASONS = (2014, 2021)
VAL_SEASONS = (2022, 2023)
TEST_SEASONS = (2024, 2025)

# Updated feature lists with new FastF1 features
NUMERIC_FEATURES = [
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
    # Pit stop features
    "total_stops",
    "mean_stop_duration",
    # Circuit features
    "circuit_lat",
    "circuit_lng",
    # Rolling performance features
    "driver_recent_avg_finish",
    "driver_circuit_avg_finish",
    "driver_avg_positions_gained",
    "constructor_recent_avg_finish",
    # Form and head-to-head
    "driver_form_trend",
    "gap_to_teammate_quali",
    # Weather features
    "track_temp",
    "air_temp",
    "humidity",
    "is_wet_race",
    "wind_speed",
    # Tyre strategy features
    "num_stints",
    "num_compounds_used",
    "avg_stint_length",
    "total_tyre_laps",
]

CATEGORICAL_FEATURES = [
    "driver_id",
    "constructor_id",
    "circuit_id",
    "primary_compound",  # tyre compound
]

TARGET_REGRESSION = "finish_position"
TARGETS_CLASSIFICATION = ["is_podium", "is_top_10", "is_dnf"]


def _temporal_split(df: pd.DataFrame):
    train = df[(df["season"] >= TRAIN_SEASONS[0]) & (df["season"] <= TRAIN_SEASONS[1])]
    val = df[(df["season"] >= VAL_SEASONS[0]) & (df["season"] <= VAL_SEASONS[1])]
    test = df[(df["season"] >= TEST_SEASONS[0]) & (df["season"] <= TEST_SEASONS[1])]
    return train, val, test


def _get_feature_matrix(df: pd.DataFrame, numeric_cols: list, cat_cols: list, fit_transform: bool, ct=None):
    for c in numeric_cols:
        if c not in df.columns:
            df = df.assign(**{c: np.nan})
    for c in cat_cols:
        if c not in df.columns:
            df = df.assign(**{c: ""})
    X_num = df[numeric_cols].fillna(df[numeric_cols].median() if fit_transform else 0)
    X_cat = df[cat_cols].astype(str).fillna("__missing__")
    if ct is None:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_num = scaler.fit_transform(X_num) if fit_transform else scaler.transform(X_num)
        from sklearn.preprocessing import LabelEncoder
        encoders = {}
        X_cat_enc = np.zeros((len(df), len(cat_cols)))
        for i, col in enumerate(cat_cols):
            if fit_transform:
                le = LabelEncoder()
                X_cat_enc[:, i] = le.fit_transform(X_cat[col].astype(str))
                encoders[col] = le
            else:
                le = encoders.get(col)
                if le is not None:
                    unk = np.array([-1] * len(df))
                    for lbl in le.classes_:
                        unk[X_cat[col].astype(str) == lbl] = le.transform([lbl])[0]
                    X_cat_enc[:, i] = unk
                else:
                    X_cat_enc[:, i] = 0
        X = np.hstack([X_num, X_cat_enc])
        return X, (scaler, encoders) if fit_transform else (None, None)
    return np.hstack([X_num, X_cat_enc]), (None, None)


def _prepare_xy(df: pd.DataFrame, target_reg: str, target_cls: str, numeric_cols: list, cat_cols: list):
    available = [c for c in numeric_cols if c in df.columns]
    X_num = df[available].copy()
    for c in numeric_cols:
        if c not in X_num.columns:
            X_num[c] = np.nan
    X_num = X_num[numeric_cols].fillna(0)
    X_cat = df[[c for c in cat_cols if c in df.columns]].copy()
    for c in cat_cols:
        if c not in X_cat.columns:
            X_cat[c] = ""
    X_cat = X_cat[cat_cols].astype(str).fillna("__missing__")
    y_reg = df[target_reg] if target_reg in df.columns else None
    y_cls = df[target_cls] if target_cls in df.columns else None
    return X_num, X_cat, y_reg, y_cls


def _build_preprocessor(X_num: pd.DataFrame, X_cat: pd.DataFrame, numeric_cols: list, cat_cols: list):
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import LabelEncoder
    X_num = X_num[[c for c in numeric_cols if c in X_num.columns]].reindex(columns=numeric_cols).fillna(0)
    X_cat = X_cat[[c for c in cat_cols if c in X_cat.columns]].reindex(columns=cat_cols).astype(str).fillna("__missing__")
    scaler = StandardScaler()
    X_num_s = scaler.fit_transform(X_num)
    encoders = {}
    X_cat_enc = np.zeros((X_cat.shape[0], len(cat_cols)))
    for i, col in enumerate(cat_cols):
        le = LabelEncoder()
        X_cat_enc[:, i] = le.fit_transform(X_cat[col].astype(str))
        encoders[col] = le
    X = np.hstack([X_num_s, X_cat_enc])
    return X, scaler, encoders, numeric_cols, cat_cols


def _apply_preprocessor(X_num: pd.DataFrame, X_cat: pd.DataFrame, scaler, encoders: dict, numeric_cols: list, cat_cols: list):
    X_num = X_num[[c for c in numeric_cols if c in X_num.columns]].reindex(columns=numeric_cols).fillna(0)
    X_cat = X_cat[[c for c in cat_cols if c in X_cat.columns]].reindex(columns=cat_cols).astype(str).fillna("__missing__")
    X_num_s = scaler.transform(X_num)
    X_cat_enc = np.zeros((X_cat.shape[0], len(cat_cols)))
    for i, col in enumerate(cat_cols):
        le = encoders.get(col)
        if le is None:
            X_cat_enc[:, i] = -1
            continue
        unk = np.array([-1] * len(X_cat))
        for lbl in le.classes_:
            unk[X_cat[col].astype(str) == lbl] = le.transform([lbl])[0]
        X_cat_enc[:, i] = unk
    return np.hstack([X_num_s, X_cat_enc])


def run_training(clean_dir: Path = None, output_dir: Path = None):
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_dir = clean_dir or CLEAN_DIR

    print("Building merged dataset...")
    df = build_merged_dataset(clean_dir)
    train_df, val_df, test_df = _temporal_split(df)

    numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    if not numeric_cols:
        numeric_cols = ["qualifying_position", "grid_position"]
        for c in numeric_cols:
            if c not in df.columns:
                df[c] = np.nan
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    if not cat_cols:
        cat_cols = ["driver_id", "constructor_id", "circuit_id"]

    has_targets = df[TARGET_REGRESSION].notna().any() if TARGET_REGRESSION in df.columns else False
    if not has_targets:
        report = {
            "status": "no_targets",
            "message": "results.csv not found or finish_position empty. Run data collection and cleaning to produce results_clean/results.csv, then re-run training.",
            "rows_merged": len(df),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
        }
        with open(output_dir / "evaluation_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print(report["message"])
        return report

    train_df = train_df.dropna(subset=[TARGET_REGRESSION])
    if len(train_df) == 0:
        report = {"status": "no_train_targets", "message": "No training rows with finish_position."}
        with open(output_dir / "evaluation_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print(report["message"])
        return report

    X_num_tr, X_cat_tr, y_reg_tr, _ = _prepare_xy(train_df, TARGET_REGRESSION, "is_podium", numeric_cols, cat_cols)
    X_num_va, X_cat_va, y_reg_va, _ = _prepare_xy(val_df, TARGET_REGRESSION, "is_podium", numeric_cols, cat_cols)
    X_num_te, X_cat_te, y_reg_te, _ = _prepare_xy(test_df, TARGET_REGRESSION, "is_podium", numeric_cols, cat_cols)

    X_tr, scaler, encoders, numeric_cols, cat_cols = _build_preprocessor(X_num_tr, X_cat_tr, numeric_cols, cat_cols)
    X_va = _apply_preprocessor(X_num_va, X_cat_va, scaler, encoders, numeric_cols, cat_cols) if len(X_num_va) > 0 else None
    X_te = _apply_preprocessor(X_num_te, X_cat_te, scaler, encoders, numeric_cols, cat_cols) if len(X_num_te) > 0 else None

    y_reg_tr = train_df[TARGET_REGRESSION].values
    y_reg_va = val_df[TARGET_REGRESSION].values if len(val_df) > 0 and val_df[TARGET_REGRESSION].notna().any() else None
    y_reg_te = test_df[TARGET_REGRESSION].values if len(test_df) > 0 and test_df[TARGET_REGRESSION].notna().any() else None

    # Full feature names list for importance reporting
    feature_names = numeric_cols + cat_cols

    report = {
        "temporal_split": {
            "train": f"{TRAIN_SEASONS[0]}-{TRAIN_SEASONS[1]}",
            "val": f"{VAL_SEASONS[0]}-{VAL_SEASONS[1]}",
            "test": f"{TEST_SEASONS[0]}-{TEST_SEASONS[1]}"
        },
        "data_summary": {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "num_features": len(feature_names),
            "numeric_features": numeric_cols,
            "categorical_features": cat_cols,
        },
        "regression": {},
        "classification": {},
    }

    def get_feature_importance_dict(model, feature_names):
        """Get feature importance with actual feature names."""
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            if len(importances) == len(feature_names):
                return {name: float(imp) for name, imp in zip(feature_names, importances)}
            else:
                # Fallback if lengths don't match
                return {f"feature_{i}": float(v) for i, v in enumerate(importances)}
        return {}

    def compute_position_bucket_accuracy(y_true, y_pred):
        """Compute accuracy for position buckets: P1-3, P4-10, P11-20."""
        def bucket(pos):
            if pos <= 3:
                return "P1-3"
            elif pos <= 10:
                return "P4-10"
            else:
                return "P11-20"

        y_true_buckets = [bucket(p) for p in y_true]
        y_pred_buckets = [bucket(max(1, min(20, round(p)))) for p in y_pred]

        correct = sum(1 for t, p in zip(y_true_buckets, y_pred_buckets) if t == p)
        return {
            "bucket_accuracy": float(correct / len(y_true)) if len(y_true) > 0 else 0.0,
            "confusion": {
                "P1-3": sum(1 for t, p in zip(y_true_buckets, y_pred_buckets) if t == "P1-3" and p == "P1-3") / max(1, sum(1 for t in y_true_buckets if t == "P1-3")),
                "P4-10": sum(1 for t, p in zip(y_true_buckets, y_pred_buckets) if t == "P4-10" and p == "P4-10") / max(1, sum(1 for t in y_true_buckets if t == "P4-10")),
                "P11-20": sum(1 for t, p in zip(y_true_buckets, y_pred_buckets) if t == "P11-20" and p == "P11-20") / max(1, sum(1 for t in y_true_buckets if t == "P11-20")),
            }
        }

    # Regression: finish_position
    print("Training regression models...")
    for name, model in [
        ("ridge", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ("random_forest", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)),
        ("gradient_boosting", GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_STATE)),
    ]:
        print(f"  Training {name}...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y_reg_tr)

        pred_tr = model.predict(X_tr)
        pred_va = model.predict(X_va) if X_va is not None and y_reg_va is not None else None
        pred_te = model.predict(X_te) if X_te is not None and y_reg_te is not None else None

        report["regression"][name] = {
            "train_mae": float(mean_absolute_error(y_reg_tr, pred_tr)),
            "train_rmse": float(np.sqrt(mean_squared_error(y_reg_tr, pred_tr))),
        }

        if pred_va is not None and y_reg_va is not None:
            valid_mask = ~np.isnan(y_reg_va)
            if valid_mask.any():
                report["regression"][name]["val_mae"] = float(mean_absolute_error(y_reg_va[valid_mask], pred_va[valid_mask]))
                report["regression"][name]["val_rmse"] = float(np.sqrt(mean_squared_error(y_reg_va[valid_mask], pred_va[valid_mask])))
                if len(np.unique(y_reg_va[valid_mask])) > 1:
                    report["regression"][name]["val_spearman"] = float(spearmanr(y_reg_va[valid_mask], pred_va[valid_mask])[0])

        if pred_te is not None and y_reg_te is not None:
            valid_mask = ~np.isnan(y_reg_te)
            if valid_mask.any():
                report["regression"][name]["test_mae"] = float(mean_absolute_error(y_reg_te[valid_mask], pred_te[valid_mask]))
                report["regression"][name]["test_rmse"] = float(np.sqrt(mean_squared_error(y_reg_te[valid_mask], pred_te[valid_mask])))
                if len(np.unique(y_reg_te[valid_mask])) > 1:
                    report["regression"][name]["test_spearman"] = float(spearmanr(y_reg_te[valid_mask], pred_te[valid_mask])[0])
                # Position bucket analysis
                bucket_stats = compute_position_bucket_accuracy(y_reg_te[valid_mask], pred_te[valid_mask])
                report["regression"][name]["position_bucket_accuracy"] = bucket_stats

        # Feature importance with names
        importance = get_feature_importance_dict(model, feature_names)
        if importance:
            # Sort by importance and keep top 15
            sorted_imp = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
            report["regression"][name]["feature_importance"] = dict(sorted_imp)

        # Check for overfitting (train/val gap)
        train_mae = report["regression"][name]["train_mae"]
        val_mae = report["regression"][name].get("val_mae", train_mae)
        report["regression"][name]["train_val_gap"] = float(val_mae - train_mae)

        joblib.dump(model, output_dir / f"model_regression_{name}.joblib")

    if HAS_XGB:
        print("  Training xgboost...")
        xgb_reg = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xgb_reg.fit(X_tr, y_reg_tr)

        pred_tr = xgb_reg.predict(X_tr)
        pred_va = xgb_reg.predict(X_va) if X_va is not None and y_reg_va is not None else None
        pred_te = xgb_reg.predict(X_te) if X_te is not None and y_reg_te is not None else None

        report["regression"]["xgboost"] = {
            "train_mae": float(mean_absolute_error(y_reg_tr, pred_tr)),
            "train_rmse": float(np.sqrt(mean_squared_error(y_reg_tr, pred_tr))),
        }

        if pred_va is not None and y_reg_va is not None:
            valid_mask = ~np.isnan(y_reg_va)
            if valid_mask.any():
                report["regression"]["xgboost"]["val_mae"] = float(mean_absolute_error(y_reg_va[valid_mask], pred_va[valid_mask]))
                report["regression"]["xgboost"]["val_rmse"] = float(np.sqrt(mean_squared_error(y_reg_va[valid_mask], pred_va[valid_mask])))
                if len(np.unique(y_reg_va[valid_mask])) > 1:
                    report["regression"]["xgboost"]["val_spearman"] = float(spearmanr(y_reg_va[valid_mask], pred_va[valid_mask])[0])

        if pred_te is not None and y_reg_te is not None:
            valid_mask = ~np.isnan(y_reg_te)
            if valid_mask.any():
                report["regression"]["xgboost"]["test_mae"] = float(mean_absolute_error(y_reg_te[valid_mask], pred_te[valid_mask]))
                report["regression"]["xgboost"]["test_rmse"] = float(np.sqrt(mean_squared_error(y_reg_te[valid_mask], pred_te[valid_mask])))
                if len(np.unique(y_reg_te[valid_mask])) > 1:
                    report["regression"]["xgboost"]["test_spearman"] = float(spearmanr(y_reg_te[valid_mask], pred_te[valid_mask])[0])
                bucket_stats = compute_position_bucket_accuracy(y_reg_te[valid_mask], pred_te[valid_mask])
                report["regression"]["xgboost"]["position_bucket_accuracy"] = bucket_stats

        # Feature importance with names
        importance = get_feature_importance_dict(xgb_reg, feature_names)
        if importance:
            sorted_imp = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
            report["regression"]["xgboost"]["feature_importance"] = dict(sorted_imp)

        train_mae = report["regression"]["xgboost"]["train_mae"]
        val_mae = report["regression"]["xgboost"].get("val_mae", train_mae)
        report["regression"]["xgboost"]["train_val_gap"] = float(val_mae - train_mae)

        joblib.dump(xgb_reg, output_dir / "model_regression_xgboost.joblib")

    # Classification: is_podium
    print("Training classification models...")
    y_cls_tr = (train_df["is_podium"] == 1).astype(int).values
    y_cls_va = (val_df["is_podium"] == 1).astype(int).values if len(val_df) > 0 and "is_podium" in val_df.columns and val_df["is_podium"].notna().any() else None
    y_cls_te = (test_df["is_podium"] == 1).astype(int).values if len(test_df) > 0 and "is_podium" in test_df.columns and test_df["is_podium"].notna().any() else None
    if np.unique(y_cls_tr).size < 2:
        report["classification"]["is_podium"] = {"status": "single_class_in_train"}
    else:
        for name, model in [
            ("logistic", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
            ("random_forest", RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=RANDOM_STATE)),
            ("gradient_boosting", GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE)),
        ]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_tr, y_cls_tr)
            pred_te = model.predict(X_te) if X_te is not None and y_cls_te is not None else None
            prob_te = model.predict_proba(X_te)[:, 1] if X_te is not None and y_cls_te is not None and hasattr(model, "predict_proba") else None
            report["classification"]["is_podium"] = report["classification"].get("is_podium") or {}
            report["classification"]["is_podium"][name] = {
                "train_accuracy": float(accuracy_score(y_cls_tr, model.predict(X_tr))),
                "train_f1": float(f1_score(y_cls_tr, model.predict(X_tr), zero_division=0)),
            }
            if pred_te is not None and y_cls_te is not None:
                report["classification"]["is_podium"][name]["test_accuracy"] = float(accuracy_score(y_cls_te, pred_te))
                report["classification"]["is_podium"][name]["test_f1"] = float(f1_score(y_cls_te, pred_te, zero_division=0))
                if prob_te is not None and np.unique(y_cls_te).size > 1:
                    report["classification"]["is_podium"][name]["test_roc_auc"] = float(roc_auc_score(y_cls_te, prob_te))
                report["classification"]["is_podium"][name]["confusion_matrix_test"] = confusion_matrix(y_cls_te, pred_te).tolist()
            joblib.dump(model, output_dir / f"model_classification_podium_{name}.joblib")

        if HAS_XGB:
            xgb_cls = xgb.XGBClassifier(n_estimators=100, max_depth=6, scale_pos_weight=(y_cls_tr == 0).sum() / max((y_cls_tr == 1).sum(), 1), random_state=RANDOM_STATE)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                xgb_cls.fit(X_tr, y_cls_tr)
            pred_te = xgb_cls.predict(X_te) if X_te is not None and y_cls_te is not None else None
            prob_te = xgb_cls.predict_proba(X_te)[:, 1] if X_te is not None and y_cls_te is not None else None
            report["classification"]["is_podium"]["xgboost"] = {
                "train_accuracy": float(accuracy_score(y_cls_tr, xgb_cls.predict(X_tr))),
                "train_f1": float(f1_score(y_cls_tr, xgb_cls.predict(X_tr), zero_division=0)),
            }
            if pred_te is not None and y_cls_te is not None:
                report["classification"]["is_podium"]["xgboost"]["test_accuracy"] = float(accuracy_score(y_cls_te, pred_te))
                report["classification"]["is_podium"]["xgboost"]["test_f1"] = float(f1_score(y_cls_te, pred_te, zero_division=0))
                if prob_te is not None and np.unique(y_cls_te).size > 1:
                    report["classification"]["is_podium"]["xgboost"]["test_roc_auc"] = float(roc_auc_score(y_cls_te, prob_te))
                report["classification"]["is_podium"]["xgboost"]["confusion_matrix_test"] = confusion_matrix(y_cls_te, pred_te).tolist()
            joblib.dump(xgb_cls, output_dir / "model_classification_podium_xgboost.joblib")

    # Classification: is_top_10
    y10_tr = (train_df["is_top_10"] == 1).astype(int).values if "is_top_10" in train_df.columns else None
    y10_te = (test_df["is_top_10"] == 1).astype(int).values if len(test_df) > 0 and "is_top_10" in test_df.columns and test_df["is_top_10"].notna().any() else None
    if y10_tr is not None and np.unique(y10_tr).size >= 2 and y10_te is not None and X_te is not None:
        report["classification"]["is_top_10"] = {}
        for name, model in [
            ("logistic", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
            ("random_forest", RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=RANDOM_STATE)),
            ("gradient_boosting", GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE)),
        ]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_tr, y10_tr)
            pred_te = model.predict(X_te)
            prob_te = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None
            report["classification"]["is_top_10"][name] = {
                "train_accuracy": float(accuracy_score(y10_tr, model.predict(X_tr))),
                "train_f1": float(f1_score(y10_tr, model.predict(X_tr), zero_division=0)),
                "test_accuracy": float(accuracy_score(y10_te, pred_te)),
                "test_f1": float(f1_score(y10_te, pred_te, zero_division=0)),
            }
            if prob_te is not None and np.unique(y10_te).size > 1:
                report["classification"]["is_top_10"][name]["test_roc_auc"] = float(roc_auc_score(y10_te, prob_te))
            joblib.dump(model, output_dir / f"model_classification_top10_{name}.joblib")
        if HAS_XGB:
            xgb_10 = xgb.XGBClassifier(n_estimators=100, max_depth=6, scale_pos_weight=(y10_tr == 0).sum() / max((y10_tr == 1).sum(), 1), random_state=RANDOM_STATE)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                xgb_10.fit(X_tr, y10_tr)
            report["classification"]["is_top_10"]["xgboost"] = {
                "train_accuracy": float(accuracy_score(y10_tr, xgb_10.predict(X_tr))),
                "train_f1": float(f1_score(y10_tr, xgb_10.predict(X_tr), zero_division=0)),
                "test_accuracy": float(accuracy_score(y10_te, xgb_10.predict(X_te))),
                "test_f1": float(f1_score(y10_te, xgb_10.predict(X_te), zero_division=0)),
            }
            if np.unique(y10_te).size > 1:
                report["classification"]["is_top_10"]["xgboost"]["test_roc_auc"] = float(roc_auc_score(y10_te, xgb_10.predict_proba(X_te)[:, 1]))
            joblib.dump(xgb_10, output_dir / "model_classification_top10_xgboost.joblib")
    else:
        report["classification"]["is_top_10"] = {"status": "skipped", "reason": "insufficient_top10_labels"}

    # Best model selection and justification
    print("Analyzing results and selecting best model...")
    reg = report.get("regression", {})
    best_reg_name = None
    best_reg_mae = float("inf")
    best_reg_spearman = 0.0

    for name, data in reg.items():
        mae = data.get("test_mae") or data.get("val_mae") or data.get("train_mae")
        if mae is not None and mae < best_reg_mae:
            best_reg_mae = mae
            best_reg_name = name
            best_reg_spearman = data.get("test_spearman") or data.get("val_spearman") or 0.0

    # Compute train/val gap for overfitting detection
    best_model_data = reg.get(best_reg_name, {})
    train_val_gap = best_model_data.get("train_val_gap", 0.0)

    report["best_model"] = {
        "regression": best_reg_name,
        "regression_mae": best_reg_mae,
        "regression_spearman": best_reg_spearman,
        "train_val_gap": train_val_gap,
        "regression_justification": (
            f"Selected {best_reg_name} for deployment: lowest test MAE ({best_reg_mae:.4f}) among Ridge, Random Forest, "
            "Gradient Boosting, XGBoost. Tree-based models capture non-linear driver/constructor effects. "
            f"Spearman correlation: {best_reg_spearman:.4f}. Train/Val gap: {train_val_gap:.4f}."
        ),
        "classification_podium": "xgboost" if "xgboost" in report.get("classification", {}).get("is_podium", {}) else "random_forest",
    }

    # Success criteria evaluation
    success_criteria = {
        "test_mae_under_2": best_reg_mae < 2.0 if best_reg_mae < float("inf") else False,
        "spearman_over_085": best_reg_spearman > 0.85 if best_reg_spearman else False,
        "no_overfitting": abs(train_val_gap) < 0.5,
    }
    success_criteria["all_passed"] = all(success_criteria.values())

    report["success_criteria"] = {
        "test_mae_under_2": {
            "target": "< 2.0",
            "actual": f"{best_reg_mae:.4f}" if best_reg_mae < float("inf") else "N/A",
            "passed": success_criteria["test_mae_under_2"],
        },
        "spearman_over_085": {
            "target": "> 0.85",
            "actual": f"{best_reg_spearman:.4f}" if best_reg_spearman else "N/A",
            "passed": success_criteria["spearman_over_085"],
        },
        "no_overfitting": {
            "target": "train_val_gap < 0.5",
            "actual": f"{train_val_gap:.4f}",
            "passed": success_criteria["no_overfitting"],
        },
        "all_passed": success_criteria["all_passed"],
    }

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Best regression model: {best_reg_name}")
    print(f"  Test MAE: {best_reg_mae:.4f}")
    print(f"  Test Spearman: {best_reg_spearman:.4f}")
    print(f"  Train/Val gap: {train_val_gap:.4f}")
    print("\nSuccess Criteria:")
    for key, val in report["success_criteria"].items():
        if key != "all_passed":
            status = "PASS" if val["passed"] else "FAIL"
            print(f"  {key}: {val['actual']} (target: {val['target']}) [{status}]")
    print(f"\nAll criteria passed: {success_criteria['all_passed']}")
    print("=" * 60)

    # Save preprocessor and feature names
    joblib.dump(
        {
            "scaler": scaler,
            "encoders": encoders,
            "numeric_cols": numeric_cols,
            "cat_cols": cat_cols,
            "feature_names": feature_names,
        },
        output_dir / "preprocessor.joblib",
    )

    with open(output_dir / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nTraining complete. Report saved to: {output_dir / 'evaluation_report.json'}")
    return report


if __name__ == "__main__":
    run_training()
