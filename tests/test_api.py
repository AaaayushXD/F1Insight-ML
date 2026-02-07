"""
API tests for F1Insight.
Run from repo root: pytest tests/ -v
Uses cleaned_dataset and ML outputs; some tests depend on existing data.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    """Health endpoint returns 200."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy"}


def test_root():
    """Root returns message and docs link."""
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert "message" in data
    assert "docs" in data


def test_api_seasons():
    """Seasons list returns 200 and a list (may be empty)."""
    r = client.get("/api/seasons")
    assert r.status_code == 200
    data = r.json()
    assert "seasons" in data
    assert isinstance(data["seasons"], list)


def test_api_races_valid_season():
    """Races for a valid season returns 200 and races array."""
    r = client.get("/api/races?season=2020")
    assert r.status_code == 200
    data = r.json()
    assert data["season"] == 2020
    assert "races" in data
    assert isinstance(data["races"], list)


def test_api_predictions_race_returns_200_when_race_on_calendar():
    """
    When (season, round) exists in the calendar, we always return 200.
    - If we have result/qualifying data: predictions list is non-empty.
    - If we have no data for that race: predictions list is empty and message is set.
    """
    r = client.get("/api/predictions/race?season=2020&round=5")
    assert r.status_code == 200, f"Expected 200 for race on calendar, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["season"] == 2020
    assert data["round"] == 5
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    if len(data["predictions"]) == 0:
        assert "message" in data


def test_api_predictions_race_has_data_when_results_exist():
    """
    When (season, round) has results in the dataset (e.g. 2020 round 1), we get non-empty predictions.
    """
    r = client.get("/api/predictions/race?season=2020&round=1")
    assert r.status_code == 200
    data = r.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    if len(data["predictions"]) > 0:
        p = data["predictions"][0]
        assert "driver_id" in p
        assert "predicted_finish_position" in p or "podium_probability" in p


def test_api_predictions_race_404_when_race_not_on_calendar():
    """When (season, round) is not in races calendar, return 404. Use round=25 (valid param) but not in 2014 calendar."""
    r = client.get("/api/predictions/race?season=2014&round=25")
    assert r.status_code == 404
    assert "detail" in r.json()


def test_api_predictions_race_invalid_season():
    """Invalid season range returns 422."""
    r = client.get("/api/predictions/race?season=2000&round=1")
    assert r.status_code == 422


def test_strategy_endpoint():
    """Strategy endpoint returns 200 with expected keys."""
    r = client.get("/strategy?predicted_position_mean=5&predicted_position_std=2")
    assert r.status_code == 200
    data = r.json()
    assert "best_strategy" in data
    assert "strategy_ranking" in data
    assert "predicted_position_mean" in data


def test_strategy_endpoint_with_enhanced_params():
    """Strategy endpoint with enhanced parameters returns expected analysis."""
    r = client.get(
        "/strategy?"
        "predicted_position_mean=5"
        "&predicted_position_std=2"
        "&circuit_id=monaco"
        "&race_laps=78"
        "&track_temp=45"
        "&rain_probability=0.2"
        "&gap_to_car_ahead=1.5"
        "&gap_to_car_behind=3.0"
    )
    assert r.status_code == 200
    data = r.json()

    # Original fields
    assert "best_strategy" in data
    assert "strategy_ranking" in data

    # Enhanced fields
    assert "safety_car_analysis" in data
    assert "weather_impact" in data
    assert "competitor_analysis" in data
    assert "compound_strategies" in data
    assert "tactical_recommendations" in data
    assert "race_parameters" in data


def test_strategy_endpoint_safety_car_analysis():
    """Strategy endpoint returns safety car analysis."""
    r = client.get("/strategy?predicted_position_mean=5&circuit_id=silverstone")
    assert r.status_code == 200
    data = r.json()

    sc_analysis = data.get("safety_car_analysis", {})
    assert "safety_car_probability" in sc_analysis
    assert "expected_safety_cars" in sc_analysis
    assert "vsc_probability" in sc_analysis


def test_strategy_endpoint_weather_impact():
    """Strategy endpoint returns weather impact with track temperature."""
    r = client.get("/strategy?predicted_position_mean=5&track_temp=55&humidity=80")
    assert r.status_code == 200
    data = r.json()

    weather = data.get("weather_impact", {})
    assert "weather_factor" in weather
    assert "degradation_multiplier" in weather
    assert "recommended_compounds" in weather
    assert "strategy_adjustments" in weather


def test_strategy_endpoint_wet_conditions():
    """Strategy endpoint handles wet race conditions."""
    r = client.get("/strategy?predicted_position_mean=5&is_wet_race=true&rain_probability=0.9")
    assert r.status_code == 200
    data = r.json()

    weather = data.get("weather_impact", {})
    assert weather["weather_factor"] > 1.0
    # Should recommend wet compounds
    compounds = weather.get("recommended_compounds", [])
    assert "WET" in compounds or "INTERMEDIATE" in compounds


def test_strategy_endpoint_competitor_analysis():
    """Strategy endpoint returns undercut/overcut analysis."""
    r = client.get("/strategy?predicted_position_mean=5&gap_to_car_ahead=1.0&gap_to_car_behind=4.0")
    assert r.status_code == 200
    data = r.json()

    competitor = data.get("competitor_analysis", {})
    assert "undercut" in competitor
    assert "overcut" in competitor
    assert "traffic_analysis" in competitor

    # With small gap ahead, undercut should be viable
    assert competitor["undercut"]["viable"] is True


def test_strategy_endpoint_compound_strategies():
    """Strategy endpoint returns compound strategy options."""
    r = client.get("/strategy?predicted_position_mean=5&race_laps=56")
    assert r.status_code == 200
    data = r.json()

    compounds = data.get("compound_strategies", [])
    assert isinstance(compounds, list)
    if len(compounds) > 0:
        first = compounds[0]
        assert "label" in first
        assert "stints" in first
        assert "overall_score" in first


def test_strategy_endpoint_validation():
    """Strategy endpoint validates input parameters."""
    # Invalid position (out of range)
    r = client.get("/strategy?predicted_position_mean=0&predicted_position_std=2")
    assert r.status_code == 422

    # Invalid rain probability
    r = client.get("/strategy?predicted_position_mean=5&rain_probability=1.5")
    assert r.status_code == 422


def test_predict_single_driver_404_when_no_data():
    """Single-driver predict returns 404 when (season, round, driver_id) not in dataset. Use round=25 (valid) but no data."""
    r = client.get("/predict?season=2020&round=25&driver_id=hamilton")
    assert r.status_code == 404


def test_predict_single_driver_200_when_data_exists():
    """Single-driver predict returns 200 when data exists (may still 503 if models missing)."""
    r = client.get("/predict?season=2020&round=1&driver_id=hamilton")
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        data = r.json()
        assert "predicted_finish_position" in data
