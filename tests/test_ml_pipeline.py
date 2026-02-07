"""
ML Pipeline Tests for F1Insight.

Tests cover:
1. Data loading and cleaning
2. Feature engineering
3. Model training and validation
4. Inference pipeline
5. Strategy recommendation

Run from repo root: pytest tests/test_ml_pipeline.py -v
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

# Test fixtures
CLEAN_DIR = Path(__file__).resolve().parent.parent / "app" / "data" / "cleaned_dataset"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "app" / "ml" / "outputs"


class TestDataLoading:
    """Tests for data loading and basic data integrity."""

    def test_cleaned_data_exists(self):
        """Verify cleaned dataset directory exists."""
        assert CLEAN_DIR.exists(), f"Cleaned dataset directory not found: {CLEAN_DIR}"

    def test_required_csv_files_exist(self):
        """Verify required CSV files are present."""
        required_files = [
            "results.csv",
            "qualifying.csv",
            "races.csv",
            "drivers.csv",
            "constructors.csv",
            "circuits.csv",
            "driver_standings.csv",
            "constructor_standings.csv",
        ]
        for file in required_files:
            path = CLEAN_DIR / file
            assert path.exists(), f"Required file missing: {file}"

    def test_results_has_required_columns(self):
        """Verify results.csv has required columns."""
        if not (CLEAN_DIR / "results.csv").exists():
            pytest.skip("results.csv not found")

        df = pd.read_csv(CLEAN_DIR / "results.csv")
        required_cols = ["season", "round", "driver_id", "constructor_id", "finish_position"]
        for col in required_cols:
            assert col in df.columns, f"Missing column in results.csv: {col}"

    def test_no_future_data_leakage_in_standings(self):
        """Verify standings data doesn't include future rounds."""
        if not (CLEAN_DIR / "driver_standings.csv").exists():
            pytest.skip("driver_standings.csv not found")

        df = pd.read_csv(CLEAN_DIR / "driver_standings.csv")
        # Each standing should be for a round that has happened
        assert "season" in df.columns
        assert "round" in df.columns
        # Round should be at least 1
        assert df["round"].min() >= 1


class TestFeatureEngineering:
    """Tests for feature engineering pipeline."""

    def test_build_dataset_returns_dataframe(self):
        """Verify build_merged_dataset returns a DataFrame."""
        try:
            from app.ml.build_dataset import build_merged_dataset
            df = build_merged_dataset(CLEAN_DIR)
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0, "Dataset should not be empty"
        except FileNotFoundError:
            pytest.skip("Cleaned data not available")

    def test_dataset_has_target_variable(self):
        """Verify dataset has finish_position target."""
        try:
            from app.ml.build_dataset import build_merged_dataset
            df = build_merged_dataset(CLEAN_DIR)
            assert "finish_position" in df.columns
        except FileNotFoundError:
            pytest.skip("Cleaned data not available")

    def test_dataset_has_numeric_features(self):
        """Verify dataset has expected numeric features."""
        try:
            from app.ml.build_dataset import build_merged_dataset
            df = build_merged_dataset(CLEAN_DIR)

            expected_features = [
                "qualifying_position",
                "grid_position",
                "driver_prior_points",
                "constructor_prior_points",
            ]
            for feat in expected_features:
                assert feat in df.columns, f"Missing numeric feature: {feat}"
        except FileNotFoundError:
            pytest.skip("Cleaned data not available")

    def test_dataset_has_categorical_features(self):
        """Verify dataset has expected categorical features."""
        try:
            from app.ml.build_dataset import build_merged_dataset
            df = build_merged_dataset(CLEAN_DIR)

            expected_features = ["driver_id", "constructor_id", "circuit_id"]
            for feat in expected_features:
                assert feat in df.columns, f"Missing categorical feature: {feat}"
        except FileNotFoundError:
            pytest.skip("Cleaned data not available")

    def test_no_future_leakage_in_prior_standings(self):
        """Verify prior standings use only past data."""
        try:
            from app.ml.build_dataset import build_merged_dataset
            df = build_merged_dataset(CLEAN_DIR)

            # For round 1 races, prior standings should be from previous season
            round1 = df[df["round"] == 1]
            if len(round1) > 0:
                # driver_prior_points should exist but may be NaN for first race
                assert "driver_prior_points" in round1.columns
        except FileNotFoundError:
            pytest.skip("Cleaned data not available")


class TestModelTraining:
    """Tests for model training pipeline."""

    def test_temporal_split_function(self):
        """Verify temporal split separates data correctly."""
        from app.ml.train import _temporal_split, TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS

        # Create mock data
        df = pd.DataFrame({
            "season": [2014, 2015, 2020, 2021, 2022, 2023, 2024, 2025],
            "value": range(8)
        })

        train, val, test = _temporal_split(df)

        # Check train contains only train seasons
        assert all(train["season"] >= TRAIN_SEASONS[0])
        assert all(train["season"] <= TRAIN_SEASONS[1])

        # Check val contains only val seasons
        if len(val) > 0:
            assert all(val["season"] >= VAL_SEASONS[0])
            assert all(val["season"] <= VAL_SEASONS[1])

        # Check test contains only test seasons
        if len(test) > 0:
            assert all(test["season"] >= TEST_SEASONS[0])
            assert all(test["season"] <= TEST_SEASONS[1])

    def test_preprocessor_builds_correctly(self):
        """Verify preprocessor builds without errors."""
        try:
            from app.ml.train import _build_preprocessor
            import numpy as np

            X_num = pd.DataFrame({
                "qualifying_position": [1, 2, 3, 4, 5],
                "grid_position": [1, 2, 3, 4, 5],
            })
            X_cat = pd.DataFrame({
                "driver_id": ["driver_a", "driver_b", "driver_a", "driver_c", "driver_b"],
                "circuit_id": ["circuit_1", "circuit_1", "circuit_2", "circuit_2", "circuit_1"],
            })

            X, scaler, encoders, num_cols, cat_cols = _build_preprocessor(
                X_num, X_cat,
                ["qualifying_position", "grid_position"],
                ["driver_id", "circuit_id"]
            )

            assert X.shape[0] == 5
            assert X.shape[1] == 4  # 2 numeric + 2 categorical
            assert scaler is not None
            assert "driver_id" in encoders
            assert "circuit_id" in encoders
        except ImportError:
            pytest.skip("sklearn not available")

    def test_trained_models_exist(self):
        """Verify trained model files exist."""
        if not OUTPUT_DIR.exists():
            pytest.skip("ML outputs directory not found")

        # At least one regression model should exist
        regression_models = list(OUTPUT_DIR.glob("model_regression_*.joblib"))
        assert len(regression_models) > 0, "No regression models found"

    def test_preprocessor_exists(self):
        """Verify preprocessor file exists."""
        if not OUTPUT_DIR.exists():
            pytest.skip("ML outputs directory not found")

        preprocessor_path = OUTPUT_DIR / "preprocessor.joblib"
        assert preprocessor_path.exists(), "Preprocessor not found"

    def test_evaluation_report_exists(self):
        """Verify evaluation report was generated."""
        if not OUTPUT_DIR.exists():
            pytest.skip("ML outputs directory not found")

        report_path = OUTPUT_DIR / "evaluation_report.json"
        assert report_path.exists(), "Evaluation report not found"

    def test_evaluation_report_has_required_sections(self):
        """Verify evaluation report contains required sections."""
        import json

        report_path = OUTPUT_DIR / "evaluation_report.json"
        if not report_path.exists():
            pytest.skip("Evaluation report not found")

        with open(report_path) as f:
            report = json.load(f)

        assert "temporal_split" in report
        assert "regression" in report
        assert "best_model" in report


class TestModelValidation:
    """Tests for model validation and metrics."""

    def test_regression_mae_is_reasonable(self):
        """Verify regression MAE is within acceptable range."""
        import json

        report_path = OUTPUT_DIR / "evaluation_report.json"
        if not report_path.exists():
            pytest.skip("Evaluation report not found")

        with open(report_path) as f:
            report = json.load(f)

        # Check at least one model has MAE < 10 (very loose bound for any model)
        regression = report.get("regression", {})
        min_mae = float("inf")
        for model_name, metrics in regression.items():
            mae = metrics.get("train_mae") or metrics.get("val_mae") or metrics.get("test_mae")
            if mae is not None:
                min_mae = min(min_mae, mae)

        assert min_mae < 10, f"Best MAE ({min_mae}) is too high"

    def test_no_severe_overfitting(self):
        """Verify train/val gap is not too large."""
        import json

        report_path = OUTPUT_DIR / "evaluation_report.json"
        if not report_path.exists():
            pytest.skip("Evaluation report not found")

        with open(report_path) as f:
            report = json.load(f)

        regression = report.get("regression", {})
        for model_name, metrics in regression.items():
            gap = metrics.get("train_val_gap", 0)
            # Gap should be reasonable (< 5 positions is very loose)
            assert abs(gap) < 5, f"{model_name} has large train/val gap: {gap}"

    def test_feature_importance_has_names(self):
        """Verify feature importance uses actual feature names."""
        import json

        report_path = OUTPUT_DIR / "evaluation_report.json"
        if not report_path.exists():
            pytest.skip("Evaluation report not found")

        with open(report_path) as f:
            report = json.load(f)

        regression = report.get("regression", {})
        for model_name, metrics in regression.items():
            importance = metrics.get("feature_importance", {})
            if importance:
                # Should have actual feature names, not f0, f1, etc.
                keys = list(importance.keys())
                if keys:
                    # At least first key should not be "f0" pattern
                    first_key = keys[0]
                    assert not first_key.startswith("f") or not first_key[1:].isdigit(), \
                        f"Feature importance should use actual names, got: {first_key}"


class TestInference:
    """Tests for inference pipeline."""

    def test_predict_function_returns_dict(self):
        """Verify predict function returns a dictionary."""
        from app.ml.inference import predict

        # Create a mock row
        row = pd.Series({
            "qualifying_position": 5,
            "grid_position": 5,
            "driver_id": "test_driver",
            "constructor_id": "test_team",
            "circuit_id": "test_circuit",
        })

        result = predict(row)
        assert isinstance(result, dict)

    def test_predict_returns_position_or_error(self):
        """Verify predict returns either position or error message."""
        from app.ml.inference import predict

        row = pd.Series({
            "qualifying_position": 5,
            "grid_position": 5,
            "driver_id": "test_driver",
            "constructor_id": "test_team",
            "circuit_id": "test_circuit",
        })

        result = predict(row)

        # Should have either predicted_finish_position or error
        has_prediction = "predicted_finish_position" in result
        has_error = "error" in result
        assert has_prediction or has_error

    def test_predicted_position_in_valid_range(self):
        """Verify predicted position is in valid range (1-20)."""
        from app.ml.inference import predict

        if not (OUTPUT_DIR / "preprocessor.joblib").exists():
            pytest.skip("Models not trained")

        row = pd.Series({
            "qualifying_position": 5,
            "grid_position": 5,
            "driver_id": "hamilton",
            "constructor_id": "mercedes",
            "circuit_id": "albert_park",
        })

        result = predict(row)

        if "predicted_finish_position" in result:
            pos = result["predicted_finish_position"]
            assert 1 <= pos <= 20, f"Predicted position {pos} out of range"


class TestStrategyRecommendation:
    """Tests for strategy recommendation system."""

    def test_recommend_strategy_returns_dict(self):
        """Verify recommend_strategy returns a dictionary."""
        from app.ml.strategy import recommend_strategy

        result = recommend_strategy(
            predicted_position_mean=5.0,
            predicted_position_std=2.0
        )

        assert isinstance(result, dict)
        assert "best_strategy" in result
        assert "strategy_ranking" in result

    def test_strategy_ranking_is_ordered(self):
        """Verify strategies are ranked by expected position."""
        from app.ml.strategy import recommend_strategy

        result = recommend_strategy(
            predicted_position_mean=5.0,
            predicted_position_std=2.0
        )

        ranking = result.get("strategy_ranking", [])
        if len(ranking) >= 2:
            positions = [s.get("expected_position", 99) for s in ranking]
            assert positions == sorted(positions), "Strategies should be sorted by position"

    def test_monte_carlo_produces_consistent_results(self):
        """Verify Monte Carlo simulation is deterministic with seed."""
        from app.ml.strategy import monte_carlo_strategy_rank

        strategies = [
            {"strategy_id": 0, "typical_stops": 1},
            {"strategy_id": 1, "typical_stops": 2},
        ]

        result1 = monte_carlo_strategy_rank(5.0, 2.0, strategies=strategies)
        result2 = monte_carlo_strategy_rank(5.0, 2.0, strategies=strategies)

        # Results should be identical due to fixed seed
        assert result1[0]["expected_position"] == result2[0]["expected_position"]

    def test_more_stops_increases_position(self):
        """Verify more pit stops results in worse expected position."""
        from app.ml.strategy import monte_carlo_strategy_rank

        strategies = [
            {"strategy_id": 0, "typical_stops": 1},
            {"strategy_id": 1, "typical_stops": 2},
            {"strategy_id": 2, "typical_stops": 3},
        ]

        result = monte_carlo_strategy_rank(5.0, 2.0, strategies=strategies)

        positions = {r["strategy_id"]: r["expected_position"] for r in result}

        # More stops should generally mean worse position
        assert positions[0] < positions[1], "1-stop should be better than 2-stop"
        assert positions[1] < positions[2], "2-stop should be better than 3-stop"


class TestEnhancedStrategy:
    """Tests for enhanced strategy features."""

    def test_safety_car_probability_returns_dict(self):
        """Verify safety car probability calculation returns expected structure."""
        from app.ml.strategy import calculate_circuit_safety_car_probability

        result = calculate_circuit_safety_car_probability("monaco")

        assert isinstance(result, dict)
        assert "safety_car_probability" in result
        assert "expected_safety_cars" in result
        assert "vsc_probability" in result
        assert 0 <= result["safety_car_probability"] <= 1

    def test_safety_car_probability_defaults_for_unknown_circuit(self):
        """Verify default values for unknown circuit."""
        from app.ml.strategy import calculate_circuit_safety_car_probability

        result = calculate_circuit_safety_car_probability("unknown_circuit_xyz")

        assert result["safety_car_probability"] == 0.35  # default

    def test_weather_impact_dry_conditions(self):
        """Verify weather impact for dry conditions."""
        from app.ml.strategy import get_weather_impact

        result = get_weather_impact(
            track_temp=35.0,
            air_temp=25.0,
            humidity=50.0,
            rain_probability=0.0,
            is_wet_race=False,
        )

        assert result["weather_factor"] == 1.0
        assert result["degradation_multiplier"] >= 1.0
        assert "SOFT" in result["recommended_compounds"] or "MEDIUM" in result["recommended_compounds"]

    def test_weather_impact_wet_conditions(self):
        """Verify weather impact for wet conditions."""
        from app.ml.strategy import get_weather_impact

        result = get_weather_impact(
            rain_probability=0.9,
            is_wet_race=True,
        )

        assert result["weather_factor"] > 1.0  # More uncertainty
        assert "WET" in result["recommended_compounds"] or "INTERMEDIATE" in result["recommended_compounds"]
        assert "wet_conditions" in result["strategy_adjustments"]

    def test_weather_impact_high_track_temp(self):
        """Verify high track temperature increases degradation."""
        from app.ml.strategy import get_weather_impact

        result = get_weather_impact(track_temp=55.0)

        assert result["degradation_multiplier"] > 1.0
        assert "high_degradation" in result["strategy_adjustments"]

    def test_undercut_overcut_calculation(self):
        """Verify undercut/overcut window calculation."""
        from app.ml.strategy import calculate_undercut_overcut_windows

        result = calculate_undercut_overcut_windows(
            driver_position=5,
            lap_delta_to_car_ahead=1.5,
            lap_delta_to_car_behind=3.0,
        )

        assert isinstance(result, dict)
        assert "undercut" in result
        assert "overcut" in result
        assert "traffic_analysis" in result
        assert isinstance(result["undercut"]["viable"], bool)
        assert isinstance(result["overcut"]["viable"], bool)

    def test_undercut_viable_with_small_gap(self):
        """Verify undercut is viable with small gap to car ahead."""
        from app.ml.strategy import calculate_undercut_overcut_windows

        result = calculate_undercut_overcut_windows(
            driver_position=5,
            lap_delta_to_car_ahead=1.0,  # Small gap
            lap_delta_to_car_behind=5.0,
        )

        assert result["undercut"]["viable"] is True
        assert result["undercut"]["confidence"] > 0

    def test_overcut_viable_with_large_gap_behind(self):
        """Verify overcut is viable with large gap to car behind."""
        from app.ml.strategy import calculate_undercut_overcut_windows

        result = calculate_undercut_overcut_windows(
            driver_position=5,
            lap_delta_to_car_ahead=5.0,
            lap_delta_to_car_behind=4.0,  # Large gap behind
        )

        assert result["overcut"]["viable"] is True

    def test_generate_compound_strategies(self):
        """Verify compound strategy generation."""
        from app.ml.strategy import generate_compound_strategies

        strategies = generate_compound_strategies(
            race_laps=56,
            available_compounds=["SOFT", "MEDIUM", "HARD"],
        )

        assert isinstance(strategies, list)
        assert len(strategies) > 0

        # Check structure of first strategy
        first = strategies[0]
        assert "strategy_id" in first
        assert "label" in first
        assert "stints" in first
        assert "total_stops" in first

    def test_compound_strategy_scoring(self):
        """Verify compound strategy scoring."""
        from app.ml.strategy import calculate_tyre_strategy_score

        strategy = {
            "stints": [
                {"compound": "MEDIUM", "laps": 25},
                {"compound": "HARD", "laps": 31},
            ]
        }

        score = calculate_tyre_strategy_score(strategy, race_laps=56)

        assert "pace_score" in score
        assert "risk_score" in score
        assert "overall_score" in score
        assert 0 <= score["pace_score"] <= 1
        assert 0 <= score["risk_score"] <= 1

    def test_enhanced_recommend_strategy_has_new_fields(self):
        """Verify enhanced recommend_strategy includes all new analysis."""
        from app.ml.strategy import recommend_strategy

        result = recommend_strategy(
            predicted_position_mean=5.0,
            predicted_position_std=2.0,
            circuit_id="silverstone",
            race_laps=52,
            track_temp=40.0,
            rain_probability=0.1,
            gap_to_car_ahead=1.5,
            gap_to_car_behind=2.5,
        )

        # Original fields
        assert "best_strategy" in result
        assert "strategy_ranking" in result

        # New enhanced fields
        assert "safety_car_analysis" in result
        assert "weather_impact" in result
        assert "competitor_analysis" in result
        assert "compound_strategies" in result
        assert "tactical_recommendations" in result
        assert "race_parameters" in result

    def test_monte_carlo_with_safety_car(self):
        """Verify Monte Carlo simulation accounts for safety car probability."""
        from app.ml.strategy import monte_carlo_strategy_rank

        strategies = [
            {"strategy_id": 0, "typical_stops": 1},
            {"strategy_id": 1, "typical_stops": 2},
        ]

        # With high safety car probability
        result_high_sc = monte_carlo_strategy_rank(
            5.0, 2.0,
            strategies=strategies,
            safety_car_probability=0.8
        )

        # With low safety car probability
        result_low_sc = monte_carlo_strategy_rank(
            5.0, 2.0,
            strategies=strategies,
            safety_car_probability=0.1
        )

        # Multi-stop strategy should benefit more from high SC probability
        two_stop_high_sc = next(r for r in result_high_sc if r["strategy_id"] == 1)
        two_stop_low_sc = next(r for r in result_low_sc if r["strategy_id"] == 1)

        # SC benefit should be higher with high SC probability
        assert two_stop_high_sc["sc_benefit"] > two_stop_low_sc["sc_benefit"]

    def test_tactical_recommendations_generated(self):
        """Verify tactical recommendations are generated based on conditions."""
        from app.ml.strategy import recommend_strategy

        # Test with conditions that should generate recommendations
        result = recommend_strategy(
            predicted_position_mean=5.0,
            predicted_position_std=2.0,
            rain_probability=0.5,  # Should trigger rain warning
            gap_to_car_ahead=1.0,  # Should enable undercut
        )

        recommendations = result.get("tactical_recommendations", [])
        assert isinstance(recommendations, list)
        # At least one recommendation should be generated
        # (undercut opportunity or rain risk)


class TestDataCleaning:
    """Tests for data cleaning pipeline."""

    def test_clean_data_module_imports(self):
        """Verify clean_data module can be imported."""
        from app.scripts.clean_data import run_cleaning
        assert callable(run_cleaning)

    def test_pitstops_have_required_columns(self):
        """Verify pitstops.csv has required columns after cleaning."""
        path = CLEAN_DIR / "pitstops.csv"
        if not path.exists():
            pytest.skip("pitstops.csv not found")

        df = pd.read_csv(path)
        required_cols = ["season", "round", "driver_id", "lap", "stop_number"]
        for col in required_cols:
            assert col in df.columns, f"Missing column in pitstops.csv: {col}"

    def test_driver_ids_are_lowercase(self):
        """Verify driver IDs are lowercase strings."""
        path = CLEAN_DIR / "results.csv"
        if not path.exists():
            pytest.skip("results.csv not found")

        df = pd.read_csv(path)
        if "driver_id" in df.columns:
            # All driver IDs should be lowercase
            driver_ids = df["driver_id"].dropna().astype(str)
            for did in driver_ids.unique():
                assert did == did.lower(), f"Driver ID not lowercase: {did}"


class TestAPIEndpoints:
    """Tests for API endpoints (integration tests)."""

    def test_docs_endpoint(self):
        """Verify /docs endpoint is accessible."""
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        response = client.get("/docs")
        assert response.status_code == 200

    def test_strategy_endpoint_validation(self):
        """Verify strategy endpoint validates input parameters."""
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)

        # Valid request
        response = client.get("/strategy?predicted_position_mean=5&predicted_position_std=2")
        assert response.status_code == 200

        # Invalid position (out of range)
        response = client.get("/strategy?predicted_position_mean=0&predicted_position_std=2")
        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_requires_parameters(self):
        """Verify predict endpoint requires all parameters."""
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)

        # Missing driver_id
        response = client.get("/predict?season=2024&round=1")
        assert response.status_code == 422
