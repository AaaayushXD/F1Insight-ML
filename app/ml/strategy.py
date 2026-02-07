"""
F1Insight strategy assistance: tyre strategy clustering and Monte Carlo simulation.
Uses historical pit-stop patterns and model predictions to rank strategies.
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
DEFAULT_PIT_LOSS_SEC = 22.0  # typical pit loss (seconds)
DEFAULT_N_SIMULATIONS = 2000
DEFAULT_DEGRADATION_STD = 0.3  # tyre degradation uncertainty (positions)
DEFAULT_TRAFFIC_LOSS_STD = 0.5  # traffic loss uncertainty (positions)


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
            {"strategy_id": 0, "label": "unknown", "typical_stops": 1, "typical_lap_of_stop": 25, "count": 0},
        ]
    out = []
    for i in range(n_clusters):
        mask = labels == i
        sub = agg.loc[mask]
        out.append({
            "strategy_id": int(i),
            "label": f"strategy_{i}",
            "typical_stops": float(sub["num_stops"].median()),
            "typical_lap_of_stop": float(sub["mean_lap_of_stop"].median()),
            "mean_duration_seconds": float(sub["mean_duration_seconds"].median()) if sub["mean_duration_seconds"].notna().any() else None,
            "count": int(mask.sum()),
        })
    return out


def monte_carlo_strategy_rank(
    predicted_position_mean: float,
    predicted_position_std: float,
    pit_loss_sec: float = DEFAULT_PIT_LOSS_SEC,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    strategies: Optional[List[Dict[str, Any]]] = None,
    degradation_std: float = DEFAULT_DEGRADATION_STD,
    traffic_loss_std: float = DEFAULT_TRAFFIC_LOSS_STD,
) -> List[Dict[str, Any]]:
    """
    Simulate race outcomes with uncertainty on finishing position; optionally factor in pit strategy.
    Position is simulated as N(mean, std). Adds degradation and traffic loss uncertainty (no telemetry).
    If strategies are given, each strategy gets an additive penalty proportional to pit_loss (extra stops).
    Returns list of { "strategy_id", "expected_position", "std_position", "rank" } sorted by expected_position.
    """
    rng = np.random.default_rng(42)
    base_std = max(0.1, predicted_position_std)
    pos_samples = rng.normal(predicted_position_mean, base_std, size=n_simulations)
    deg_noise = rng.normal(0, degradation_std, size=n_simulations)
    traffic_noise = rng.normal(0, traffic_loss_std, size=n_simulations)
    pos_samples = pos_samples + deg_noise + traffic_noise
    if not strategies:
        return [{
            "strategy_id": 0,
            "expected_position": float(np.mean(pos_samples)),
            "std_position": float(np.std(pos_samples)),
            "rank": 1,
        }]
    position_per_sec = 1.0 / 1.5
    results = []
    for s in strategies:
        extra_stops = max(0, s.get("typical_stops", 1) - 1)
        penalty = extra_stops * pit_loss_sec * position_per_sec
        adjusted = pos_samples + penalty
        results.append({
            "strategy_id": s.get("strategy_id", 0),
            "label": s.get("label", ""),
            "expected_position": float(np.mean(adjusted)),
            "std_position": float(np.std(adjusted)),
            "typical_stops": s.get("typical_stops"),
        })
    results.sort(key=lambda x: x["expected_position"])
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
) -> Dict[str, Any]:
    """
    High-level recommendation: run strategy clustering, then Monte Carlo rank.
    Incorporates degradation and traffic uncertainty. Returns best strategy and full ranking.
    """
    summary = get_strategy_summary(clean_dir, n_clusters=3)
    ranking = monte_carlo_strategy_rank(
        predicted_position_mean,
        predicted_position_std,
        pit_loss_sec=pit_loss_sec,
        n_simulations=n_simulations,
        strategies=summary,
        degradation_std=degradation_std,
        traffic_loss_std=traffic_loss_std,
    )
    best = ranking[0] if ranking else {"strategy_id": 0, "expected_position": predicted_position_mean}
    return {
        "predicted_position_mean": predicted_position_mean,
        "predicted_position_std": predicted_position_std,
        "best_strategy": best,
        "strategy_ranking": ranking,
        "strategy_summary": summary,
    }
