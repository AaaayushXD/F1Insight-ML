"""
F1Insight Enhanced Strategy Recommendation System.

Features:
1. Tyre Compound Optimization - Model soft/medium/hard trade-offs with degradation curves
2. Safety Car Probability - Historical SC likelihood per circuit from incident data
3. Weather Integration - Rain probability, wet tyre strategy
4. Competitor Modeling - Undercut/overcut strategies, traffic pattern prediction

Uses KMeans clustering for strategy archetypes and Monte Carlo simulation for ranking.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

CLEAN_DIR = Path(__file__).resolve().parent.parent / "data" / "cleaned_dataset"

# Default simulation parameters
DEFAULT_PIT_LOSS_SEC = 22.0  # typical pit loss (seconds)
DEFAULT_N_SIMULATIONS = 2000
DEFAULT_DEGRADATION_STD = 0.3  # tyre degradation uncertainty (positions)
DEFAULT_TRAFFIC_LOSS_STD = 0.5  # traffic loss uncertainty (positions)

# Tyre compound characteristics
COMPOUND_CHARACTERISTICS = {
    "SOFT": {
        "pace_advantage": 0.8,  # seconds per lap faster than medium
        "degradation_rate": 0.08,  # position loss per lap due to degradation
        "optimal_stint_length": 15,  # laps before significant dropoff
        "max_stint_length": 22,  # maximum viable stint
    },
    "MEDIUM": {
        "pace_advantage": 0.0,  # baseline compound
        "degradation_rate": 0.04,
        "optimal_stint_length": 25,
        "max_stint_length": 35,
    },
    "HARD": {
        "pace_advantage": -0.5,  # slower but more durable
        "degradation_rate": 0.02,
        "optimal_stint_length": 35,
        "max_stint_length": 50,
    },
    "INTERMEDIATE": {
        "pace_advantage": -3.0,  # only for light rain
        "degradation_rate": 0.01,
        "optimal_stint_length": 40,
        "max_stint_length": 60,
    },
    "WET": {
        "pace_advantage": -6.0,  # heavy rain only
        "degradation_rate": 0.005,
        "optimal_stint_length": 50,
        "max_stint_length": 70,
    },
}

# Strategy templates for different scenarios
STRATEGY_TEMPLATES = {
    "dry_aggressive": [
        {"stint": 1, "compound": "SOFT", "laps": 18},
        {"stint": 2, "compound": "MEDIUM", "laps": 38},
    ],
    "dry_standard": [
        {"stint": 1, "compound": "MEDIUM", "laps": 25},
        {"stint": 2, "compound": "HARD", "laps": 31},
    ],
    "dry_conservative": [
        {"stint": 1, "compound": "HARD", "laps": 20},
        {"stint": 2, "compound": "MEDIUM", "laps": 18},
        {"stint": 3, "compound": "SOFT", "laps": 18},
    ],
    "wet_to_dry": [
        {"stint": 1, "compound": "INTERMEDIATE", "laps": 15},
        {"stint": 2, "compound": "MEDIUM", "laps": 25},
        {"stint": 3, "compound": "SOFT", "laps": 16},
    ],
    "mixed_conditions": [
        {"stint": 1, "compound": "INTERMEDIATE", "laps": 20},
        {"stint": 2, "compound": "SOFT", "laps": 20},
        {"stint": 3, "compound": "MEDIUM", "laps": 16},
    ],
}


def load_pitstop_aggregates(clean_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load pitstops and aggregate per (season, round, driver_id): num_stops, mean_lap_of_stop, mean_duration."""
    base = clean_dir or CLEAN_DIR
    path = base / "pitstops.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty or "lap" not in df.columns:
        return pd.DataFrame()
    agg = df.groupby(["season", "round", "driver_id"]).agg(
        num_stops=("stop_number", "max"),
        mean_lap_of_stop=("lap", "mean"),
        mean_duration_seconds=("duration_seconds", "mean"),
    ).reset_index()
    return agg


def calculate_circuit_safety_car_probability(
    circuit_id: str,
    clean_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Calculate safety car probability for a circuit based on historical incidents.

    Uses DNF data (accidents, collisions, spun off) to estimate SC probability.
    Returns probability of at least one SC, expected number of SCs, and VSC probability.
    """
    base = clean_dir or CLEAN_DIR
    results_path = base / "results.csv"
    races_path = base / "races.csv"

    if not results_path.exists() or not races_path.exists():
        return {
            "safety_car_probability": 0.35,  # default ~35% of races have SC
            "expected_safety_cars": 0.5,
            "vsc_probability": 0.25,
        }

    results = pd.read_csv(results_path)
    races = pd.read_csv(races_path)

    # Merge to get circuit info
    results = results.merge(
        races[["season", "round", "circuit_id"]],
        on=["season", "round"],
        how="left"
    )

    # Filter to specific circuit
    circuit_results = results[results["circuit_id"] == circuit_id]

    if circuit_results.empty:
        return {
            "safety_car_probability": 0.35,
            "expected_safety_cars": 0.5,
            "vsc_probability": 0.25,
        }

    # Count incident types that typically cause safety cars
    # status_id 3 = Accident, 4 = Collision, 20 = Spun off, 130 = Collision damage
    sc_causing_statuses = [3, 4, 20, 130]

    race_incidents = circuit_results.groupby(["season", "round"]).agg(
        incident_count=("status_id", lambda x: x.isin(sc_causing_statuses).sum()),
        total_dnfs=("is_dnf", "sum"),
    ).reset_index()

    total_races = len(race_incidents)
    races_with_incidents = (race_incidents["incident_count"] > 0).sum()

    # Estimate SC probability based on incidents
    sc_prob = min(0.9, races_with_incidents / max(1, total_races) * 1.2)  # incidents often cause SC

    # VSC probability (minor incidents)
    vsc_races = (race_incidents["incident_count"] == 1).sum()
    vsc_prob = vsc_races / max(1, total_races)

    # Expected number of safety cars
    avg_incidents = race_incidents["incident_count"].mean()
    expected_sc = min(2.0, avg_incidents * 0.7)  # not all incidents cause SC

    return {
        "safety_car_probability": round(sc_prob, 3),
        "expected_safety_cars": round(expected_sc, 2),
        "vsc_probability": round(vsc_prob, 3),
        "historical_races_analyzed": total_races,
        "circuit_id": circuit_id,
    }


def get_weather_impact(
    track_temp: Optional[float] = None,
    air_temp: Optional[float] = None,
    humidity: Optional[float] = None,
    rain_probability: float = 0.0,
    is_wet_race: bool = False,
) -> Dict[str, Any]:
    """
    Calculate weather impact on strategy.

    Returns:
        dict with weather_factor, recommended_compounds, and strategy_adjustments
    """
    weather_factor = 1.0  # baseline
    strategy_adjustments = []
    recommended_compounds = ["MEDIUM", "HARD", "SOFT"]

    if is_wet_race or rain_probability > 0.7:
        weather_factor = 1.5  # more uncertainty
        recommended_compounds = ["WET", "INTERMEDIATE"]
        strategy_adjustments.append("wet_conditions")
    elif rain_probability > 0.3:
        weather_factor = 1.3
        recommended_compounds = ["INTERMEDIATE", "MEDIUM", "SOFT"]
        strategy_adjustments.append("potential_rain")

    # Temperature effects on tyre degradation
    degradation_multiplier = 1.0
    if track_temp is not None:
        if track_temp > 50:
            degradation_multiplier = 1.4  # high deg, favor harder compounds
            strategy_adjustments.append("high_degradation")
        elif track_temp > 40:
            degradation_multiplier = 1.2
        elif track_temp < 25:
            degradation_multiplier = 0.8  # low deg, can push softs longer
            strategy_adjustments.append("low_degradation")

    # Humidity effects
    if humidity is not None and humidity > 80:
        strategy_adjustments.append("high_humidity")
        weather_factor *= 1.1

    return {
        "weather_factor": round(weather_factor, 2),
        "degradation_multiplier": round(degradation_multiplier, 2),
        "recommended_compounds": recommended_compounds,
        "strategy_adjustments": strategy_adjustments,
        "conditions": {
            "track_temp": track_temp,
            "air_temp": air_temp,
            "humidity": humidity,
            "rain_probability": rain_probability,
            "is_wet_race": is_wet_race,
        },
    }


def calculate_undercut_overcut_windows(
    driver_position: int,
    lap_delta_to_car_ahead: float = 1.0,
    lap_delta_to_car_behind: float = 1.0,
    pit_loss_sec: float = DEFAULT_PIT_LOSS_SEC,
    tyre_advantage_laps: int = 3,
) -> Dict[str, Any]:
    """
    Calculate undercut and overcut opportunity windows.

    Undercut: Pit earlier to gain time on fresh tyres before competitor pits.
    Overcut: Stay out and gain track position while competitors pit.

    Args:
        driver_position: Current position (1-20)
        lap_delta_to_car_ahead: Gap in seconds to car ahead
        lap_delta_to_car_behind: Gap in seconds to car behind
        pit_loss_sec: Total time lost in pit stop
        tyre_advantage_laps: Laps of fresh tyre advantage

    Returns:
        dict with undercut/overcut viability and recommended windows
    """
    # Undercut calculation
    # Fresh tyre advantage ~0.5-1.0s per lap for first few laps
    fresh_tyre_gain_per_lap = 0.7
    undercut_gain = tyre_advantage_laps * fresh_tyre_gain_per_lap

    undercut_viable = undercut_gain > lap_delta_to_car_ahead and lap_delta_to_car_ahead < 3.0
    undercut_confidence = min(1.0, undercut_gain / max(0.1, lap_delta_to_car_ahead)) if undercut_viable else 0.0

    # Overcut calculation
    # Works when traffic or tyre deg makes pitting first disadvantageous
    overcut_viable = lap_delta_to_car_behind > 2.5 and driver_position > 1
    overcut_confidence = min(1.0, lap_delta_to_car_behind / 5.0) if overcut_viable else 0.0

    # Traffic analysis
    in_traffic = lap_delta_to_car_ahead < 1.5
    clear_air_available = lap_delta_to_car_ahead > 3.0

    return {
        "undercut": {
            "viable": undercut_viable,
            "confidence": round(undercut_confidence, 2),
            "potential_gain_positions": 1 if undercut_viable else 0,
            "recommended_action": "pit_early" if undercut_viable and undercut_confidence > 0.6 else "none",
        },
        "overcut": {
            "viable": overcut_viable,
            "confidence": round(overcut_confidence, 2),
            "potential_gain_positions": 1 if overcut_viable and overcut_confidence > 0.5 else 0,
            "recommended_action": "stay_out" if overcut_viable and overcut_confidence > 0.5 else "none",
        },
        "traffic_analysis": {
            "in_traffic": in_traffic,
            "clear_air_available": clear_air_available,
            "gap_to_car_ahead": lap_delta_to_car_ahead,
            "gap_to_car_behind": lap_delta_to_car_behind,
        },
        "position": driver_position,
    }


def calculate_tyre_strategy_score(
    strategy: Dict[str, Any],
    race_laps: int = 56,
    degradation_multiplier: float = 1.0,
    safety_car_probability: float = 0.35,
) -> Dict[str, float]:
    """
    Score a tyre strategy based on compound selection, stint lengths, and conditions.

    Returns:
        dict with pace_score, risk_score, overall_score, and breakdown
    """
    stints = strategy.get("stints", [])
    if not stints:
        return {"pace_score": 0.0, "risk_score": 1.0, "overall_score": 0.0}

    pace_score = 0.0
    risk_score = 0.0
    total_laps = 0

    for stint in stints:
        compound = stint.get("compound", "MEDIUM")
        stint_laps = stint.get("laps", 20)
        total_laps += stint_laps

        chars = COMPOUND_CHARACTERISTICS.get(compound, COMPOUND_CHARACTERISTICS["MEDIUM"])

        # Pace contribution (faster compounds = higher score)
        pace_contribution = (chars["pace_advantage"] + 1) * stint_laps / race_laps
        pace_score += pace_contribution

        # Risk from pushing tyres beyond optimal
        adjusted_optimal = chars["optimal_stint_length"] / degradation_multiplier
        if stint_laps > adjusted_optimal:
            overpush = (stint_laps - adjusted_optimal) / chars["max_stint_length"]
            risk_score += overpush * 0.3

        # Risk from compound degradation rate
        risk_score += chars["degradation_rate"] * stint_laps * degradation_multiplier * 0.5

    # Penalty for not covering race distance
    if total_laps < race_laps:
        pace_score *= (total_laps / race_laps)

    # Safety car benefit for multi-stop strategies (free pit stops)
    num_stops = len(stints) - 1
    if num_stops >= 2:
        sc_benefit = safety_car_probability * 0.15 * num_stops
        pace_score += sc_benefit
        risk_score -= sc_benefit * 0.5

    # Normalize scores
    pace_score = max(0.0, min(1.0, (pace_score + 1) / 2))
    risk_score = max(0.0, min(1.0, risk_score))

    # Overall score (pace weighted higher than risk)
    overall_score = pace_score * 0.7 - risk_score * 0.3

    return {
        "pace_score": round(pace_score, 3),
        "risk_score": round(risk_score, 3),
        "overall_score": round(overall_score, 3),
        "total_stops": num_stops,
        "total_laps_covered": total_laps,
    }


def generate_compound_strategies(
    race_laps: int = 56,
    available_compounds: List[str] = None,
    degradation_multiplier: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Generate viable tyre compound strategies for a race.

    Returns list of strategy options with compound sequences and expected performance.
    """
    if available_compounds is None:
        available_compounds = ["SOFT", "MEDIUM", "HARD"]

    strategies = []
    strategy_id = 0

    # Filter to available compounds
    compounds = [c for c in available_compounds if c in COMPOUND_CHARACTERISTICS]
    if not compounds:
        compounds = ["MEDIUM"]

    # 1-stop strategies
    for c1 in compounds:
        for c2 in compounds:
            if c1 == c2:
                continue

            chars1 = COMPOUND_CHARACTERISTICS[c1]
            chars2 = COMPOUND_CHARACTERISTICS[c2]

            stint1_laps = int(chars1["optimal_stint_length"] / degradation_multiplier)
            stint2_laps = race_laps - stint1_laps

            if stint2_laps < 10 or stint2_laps > chars2["max_stint_length"] / degradation_multiplier:
                continue

            strategies.append({
                "strategy_id": strategy_id,
                "label": f"{c1[:1]}-{c2[:1]} (1-stop)",
                "stints": [
                    {"stint": 1, "compound": c1, "laps": stint1_laps},
                    {"stint": 2, "compound": c2, "laps": stint2_laps},
                ],
                "total_stops": 1,
                "compound_sequence": [c1, c2],
            })
            strategy_id += 1

    # 2-stop strategies
    for c1 in compounds:
        for c2 in compounds:
            for c3 in compounds:
                chars1 = COMPOUND_CHARACTERISTICS[c1]
                chars2 = COMPOUND_CHARACTERISTICS[c2]
                chars3 = COMPOUND_CHARACTERISTICS[c3]

                stint1_laps = int(chars1["optimal_stint_length"] * 0.8 / degradation_multiplier)
                stint2_laps = int(chars2["optimal_stint_length"] * 0.8 / degradation_multiplier)
                stint3_laps = race_laps - stint1_laps - stint2_laps

                if stint3_laps < 8 or stint3_laps > chars3["max_stint_length"] / degradation_multiplier:
                    continue

                strategies.append({
                    "strategy_id": strategy_id,
                    "label": f"{c1[:1]}-{c2[:1]}-{c3[:1]} (2-stop)",
                    "stints": [
                        {"stint": 1, "compound": c1, "laps": stint1_laps},
                        {"stint": 2, "compound": c2, "laps": stint2_laps},
                        {"stint": 3, "compound": c3, "laps": stint3_laps},
                    ],
                    "total_stops": 2,
                    "compound_sequence": [c1, c2, c3],
                })
                strategy_id += 1

    return strategies


def cluster_strategies(
    clean_dir: Optional[Path] = None,
    n_clusters: int = 3,
    random_state: int = 42,
) -> Tuple[Optional[np.ndarray], Optional[Any], Optional[Any], pd.DataFrame]:
    """
    Cluster historical pit-stop strategies by (num_stops, mean_lap_of_stop, mean_duration).
    Returns (labels, scaler, kmeans, feature_df) or (None, None, None, empty_df) if insufficient data.
    """
    if not HAS_SKLEARN:
        return None, None, None, pd.DataFrame()
    agg = load_pitstop_aggregates(clean_dir)
    if agg.empty or len(agg) < n_clusters:
        return None, None, None, agg
    X = agg[["num_stops", "mean_lap_of_stop", "mean_duration_seconds"]].fillna(0)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_s)
    return labels, scaler, kmeans, agg


def get_strategy_summary(clean_dir: Optional[Path] = None, n_clusters: int = 3) -> List[Dict[str, Any]]:
    """
    Return a summary of strategy clusters (e.g. one-stop, two-stop) with typical num_stops and lap.
    """
    labels, scaler, kmeans, agg = cluster_strategies(clean_dir, n_clusters=n_clusters)
    if labels is None or kmeans is None or agg.empty:
        return [
            {"strategy_id": 0, "label": "one-stop", "typical_stops": 1, "typical_lap_of_stop": 25, "count": 0},
            {"strategy_id": 1, "label": "two-stop", "typical_stops": 2, "typical_lap_of_stop": 20, "count": 0},
            {"strategy_id": 2, "label": "three-stop", "typical_stops": 3, "typical_lap_of_stop": 15, "count": 0},
        ]
    out = []
    for i in range(n_clusters):
        mask = labels == i
        sub = agg.loc[mask]
        typical_stops = float(sub["num_stops"].median())

        # Generate human-readable label
        if typical_stops <= 1.2:
            label = "one-stop"
        elif typical_stops <= 2.2:
            label = "two-stop"
        else:
            label = "multi-stop"

        out.append({
            "strategy_id": int(i),
            "label": label,
            "typical_stops": typical_stops,
            "typical_lap_of_stop": float(sub["mean_lap_of_stop"].median()),
            "mean_duration_seconds": float(sub["mean_duration_seconds"].median()) if sub["mean_duration_seconds"].notna().any() else None,
            "count": int(mask.sum()),
        })

    # Sort by typical stops
    out.sort(key=lambda x: x["typical_stops"])
    return out


def monte_carlo_strategy_rank(
    predicted_position_mean: float,
    predicted_position_std: float,
    pit_loss_sec: float = DEFAULT_PIT_LOSS_SEC,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    strategies: Optional[List[Dict[str, Any]]] = None,
    degradation_std: float = DEFAULT_DEGRADATION_STD,
    traffic_loss_std: float = DEFAULT_TRAFFIC_LOSS_STD,
    safety_car_probability: float = 0.35,
    weather_factor: float = 1.0,
    competitor_positions: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Enhanced Monte Carlo simulation for strategy ranking.

    Simulates race outcomes with:
    - Position uncertainty from predictions
    - Tyre degradation effects
    - Traffic/DRS considerations
    - Safety car probability (can give free pit stops)
    - Weather uncertainty
    - Competitor interactions

    Returns list of strategies ranked by expected finishing position.
    """
    rng = np.random.default_rng(42)
    base_std = max(0.1, predicted_position_std) * weather_factor

    # Base position samples
    pos_samples = rng.normal(predicted_position_mean, base_std, size=n_simulations)

    # Add uncertainty factors
    deg_noise = rng.normal(0, degradation_std * weather_factor, size=n_simulations)
    traffic_noise = rng.normal(0, traffic_loss_std, size=n_simulations)
    pos_samples = pos_samples + deg_noise + traffic_noise

    if not strategies:
        return [{
            "strategy_id": 0,
            "expected_position": float(np.mean(pos_samples)),
            "std_position": float(np.std(pos_samples)),
            "rank": 1,
        }]

    # Position loss per second in pit lane
    position_per_sec = 1.0 / 1.5

    # Simulate safety car occurrences
    sc_occurred = rng.random(n_simulations) < safety_car_probability

    results = []
    for s in strategies:
        extra_stops = max(0, s.get("typical_stops", 1) - 1)

        # Base penalty for extra pit stops
        base_penalty = extra_stops * pit_loss_sec * position_per_sec

        # Adjust penalty if safety car occurred (reduces effective pit loss)
        adjusted_penalty = np.where(
            sc_occurred & (extra_stops > 0),
            base_penalty * 0.3,  # SC reduces pit stop penalty significantly
            base_penalty
        )

        # Additional variance for multi-stop strategies
        strategy_variance = extra_stops * 0.5
        extra_noise = rng.normal(0, strategy_variance, size=n_simulations)

        adjusted = pos_samples + adjusted_penalty + extra_noise

        # Clamp to valid positions
        adjusted = np.clip(adjusted, 1, 20)

        # Calculate confidence intervals
        percentile_25 = float(np.percentile(adjusted, 25))
        percentile_75 = float(np.percentile(adjusted, 75))

        results.append({
            "strategy_id": s.get("strategy_id", 0),
            "label": s.get("label", ""),
            "expected_position": float(np.mean(adjusted)),
            "std_position": float(np.std(adjusted)),
            "typical_stops": s.get("typical_stops"),
            "best_case": float(np.min(adjusted)),
            "worst_case": float(np.max(adjusted)),
            "percentile_25": percentile_25,
            "percentile_75": percentile_75,
            "sc_benefit": float(np.mean(base_penalty - adjusted_penalty)),
        })

    # Sort by expected position
    results.sort(key=lambda x: x["expected_position"])

    # Assign ranks
    for r, row in enumerate(results, start=1):
        row["rank"] = r

    return results


def recommend_strategy(
    predicted_position_mean: float,
    predicted_position_std: float = 2.0,
    clean_dir: Optional[Path] = None,
    pit_loss_sec: float = DEFAULT_PIT_LOSS_SEC,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    degradation_std: float = DEFAULT_DEGRADATION_STD,
    traffic_loss_std: float = DEFAULT_TRAFFIC_LOSS_STD,
    # New parameters
    circuit_id: Optional[str] = None,
    race_laps: int = 56,
    track_temp: Optional[float] = None,
    air_temp: Optional[float] = None,
    humidity: Optional[float] = None,
    rain_probability: float = 0.0,
    is_wet_race: bool = False,
    gap_to_car_ahead: float = 1.0,
    gap_to_car_behind: float = 1.0,
    available_compounds: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Enhanced strategy recommendation with all new features.

    Includes:
    - Tyre compound optimization
    - Safety car probability analysis
    - Weather integration
    - Competitor modeling (undercut/overcut)

    Returns comprehensive strategy recommendation with rankings and analysis.
    """
    base = clean_dir or CLEAN_DIR

    # 1. Calculate safety car probability for circuit
    sc_analysis = calculate_circuit_safety_car_probability(
        circuit_id or "unknown",
        base
    )

    # 2. Get weather impact
    weather_impact = get_weather_impact(
        track_temp=track_temp,
        air_temp=air_temp,
        humidity=humidity,
        rain_probability=rain_probability,
        is_wet_race=is_wet_race,
    )

    # 3. Calculate undercut/overcut opportunities
    competitor_analysis = calculate_undercut_overcut_windows(
        driver_position=int(round(predicted_position_mean)),
        lap_delta_to_car_ahead=gap_to_car_ahead,
        lap_delta_to_car_behind=gap_to_car_behind,
        pit_loss_sec=pit_loss_sec,
    )

    # 4. Generate compound-specific strategies
    if available_compounds is None:
        available_compounds = weather_impact["recommended_compounds"]

    compound_strategies = generate_compound_strategies(
        race_laps=race_laps,
        available_compounds=available_compounds,
        degradation_multiplier=weather_impact["degradation_multiplier"],
    )

    # Score each compound strategy
    for cs in compound_strategies:
        score = calculate_tyre_strategy_score(
            cs,
            race_laps=race_laps,
            degradation_multiplier=weather_impact["degradation_multiplier"],
            safety_car_probability=sc_analysis["safety_car_probability"],
        )
        cs.update(score)

    # Sort compound strategies by overall score
    compound_strategies.sort(key=lambda x: x.get("overall_score", 0), reverse=True)

    # 5. Get historical strategy clusters
    summary = get_strategy_summary(base, n_clusters=3)

    # 6. Run Monte Carlo simulation for strategy ranking
    ranking = monte_carlo_strategy_rank(
        predicted_position_mean,
        predicted_position_std,
        pit_loss_sec=pit_loss_sec,
        n_simulations=n_simulations,
        strategies=summary,
        degradation_std=degradation_std,
        traffic_loss_std=traffic_loss_std,
        safety_car_probability=sc_analysis["safety_car_probability"],
        weather_factor=weather_impact["weather_factor"],
    )

    best = ranking[0] if ranking else {"strategy_id": 0, "expected_position": predicted_position_mean}

    # 7. Generate tactical recommendations
    tactical_recommendations = []

    if competitor_analysis["undercut"]["viable"]:
        tactical_recommendations.append({
            "type": "undercut_opportunity",
            "action": "Consider pitting 1-2 laps earlier than planned",
            "confidence": competitor_analysis["undercut"]["confidence"],
        })

    if competitor_analysis["overcut"]["viable"]:
        tactical_recommendations.append({
            "type": "overcut_opportunity",
            "action": "Consider staying out 2-3 laps longer",
            "confidence": competitor_analysis["overcut"]["confidence"],
        })

    if sc_analysis["safety_car_probability"] > 0.5:
        tactical_recommendations.append({
            "type": "safety_car_likely",
            "action": "Prepare for opportunistic pit stop during SC",
            "confidence": sc_analysis["safety_car_probability"],
        })

    if "high_degradation" in weather_impact["strategy_adjustments"]:
        tactical_recommendations.append({
            "type": "high_degradation",
            "action": "Favor harder compound choices",
            "confidence": 0.8,
        })

    if "potential_rain" in weather_impact["strategy_adjustments"]:
        tactical_recommendations.append({
            "type": "rain_risk",
            "action": "Keep intermediates ready, monitor conditions",
            "confidence": rain_probability,
        })

    return {
        "predicted_position_mean": predicted_position_mean,
        "predicted_position_std": predicted_position_std,
        "best_strategy": best,
        "strategy_ranking": ranking,
        "strategy_summary": summary,
        # New enhanced outputs
        "safety_car_analysis": sc_analysis,
        "weather_impact": weather_impact,
        "competitor_analysis": competitor_analysis,
        "compound_strategies": compound_strategies[:5],  # Top 5 compound strategies
        "tactical_recommendations": tactical_recommendations,
        "race_parameters": {
            "race_laps": race_laps,
            "pit_loss_sec": pit_loss_sec,
            "circuit_id": circuit_id,
        },
    }
