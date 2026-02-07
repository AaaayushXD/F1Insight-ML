# Technical Documentation: Enhanced Pit Stop Strategy Recommendation Model

## Overview

The F1Insight strategy recommendation system uses a comprehensive approach combining:
1. **KMeans Clustering**: Historical pit stop pattern analysis
2. **Monte Carlo Simulation**: Race outcome simulation with uncertainty
3. **Tyre Compound Optimization**: Degradation modeling for compound selection
4. **Safety Car Probability**: Circuit-specific incident analysis
5. **Weather Integration**: Temperature and rain impact on strategy
6. **Competitor Modeling**: Undercut/overcut opportunity analysis

## Problem Definition

### Objective
Recommend the optimal pit stop strategy for a driver given their predicted finishing position, circuit conditions, weather, and competitive situation.

### Input Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| predicted_position_mean | float | Expected finish position (1-20) |
| predicted_position_std | float | Position uncertainty |
| circuit_id | str | Circuit identifier for SC probability |
| race_laps | int | Total race laps |
| track_temp | float | Track temperature (°C) |
| rain_probability | float | Rain probability (0-1) |
| gap_to_car_ahead | float | Gap in seconds to car ahead |
| gap_to_car_behind | float | Gap in seconds to car behind |

## System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Historical     │────▶│   KMeans        │────▶│  Monte Carlo    │
│  Pit Stop Data  │     │   Clustering    │     │  Simulation     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
  - num_stops             Strategy Types:         Ranking by:
  - mean_lap_of_stop      - 1-stop               - Expected position
  - mean_duration         - 2-stop               - Position std
                          - 3-stop               - Confidence

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Safety Car     │────▶│   Weather       │────▶│   Competitor    │
│  Analysis       │     │   Integration   │     │   Modeling      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
  - SC probability        - Degradation mult     - Undercut viable
  - VSC probability       - Recommended          - Overcut viable
  - Expected SCs            compounds            - Traffic analysis
```

## 1. Tyre Compound Optimization

### Compound Characteristics

| Compound | Pace Advantage | Degradation Rate | Optimal Stint | Max Stint |
|----------|----------------|------------------|---------------|-----------|
| SOFT | +0.8s/lap | 0.08 pos/lap | 15 laps | 22 laps |
| MEDIUM | baseline | 0.04 pos/lap | 25 laps | 35 laps |
| HARD | -0.5s/lap | 0.02 pos/lap | 35 laps | 50 laps |
| INTERMEDIATE | -3.0s/lap | 0.01 pos/lap | 40 laps | 60 laps |
| WET | -6.0s/lap | 0.005 pos/lap | 50 laps | 70 laps |

### Strategy Generation Algorithm

```python
def generate_compound_strategies(
    race_laps: int = 56,
    available_compounds: List[str] = ["SOFT", "MEDIUM", "HARD"],
    degradation_multiplier: float = 1.0,
) -> List[Dict]:
    """
    Generate viable compound strategies based on race length and conditions.

    1-stop strategies: Pairs of compounds (e.g., S-M, M-H, S-H)
    2-stop strategies: Triples of compounds (e.g., S-M-S, M-S-M)
    """
```

### Strategy Scoring Formula

```
pace_score = Σ (pace_advantage + 1) × stint_laps / race_laps
risk_score = Σ overpush_penalty + degradation_penalty
overall_score = pace_score × 0.7 - risk_score × 0.3

where:
  overpush_penalty = (stint_laps - optimal) / max_stint × 0.3 if stint > optimal
  degradation_penalty = deg_rate × stint_laps × deg_multiplier × 0.5
```

### Example Strategy Options

| Strategy | Label | Stints | Pace Score | Risk Score | Overall |
|----------|-------|--------|------------|------------|---------|
| 0 | S-M (1-stop) | S:15, M:41 | 0.65 | 0.15 | 0.41 |
| 1 | M-H (1-stop) | M:25, H:31 | 0.52 | 0.10 | 0.33 |
| 2 | S-M-S (2-stop) | S:12, M:20, S:24 | 0.70 | 0.25 | 0.42 |

## 2. Safety Car Probability

### Historical Analysis Method

Safety car probability is calculated from historical incident data per circuit:

```python
def calculate_circuit_safety_car_probability(circuit_id: str) -> Dict:
    """
    Analyze historical results to estimate SC probability.

    SC-causing statuses:
    - 3: Accident
    - 4: Collision
    - 20: Spun off
    - 130: Collision damage

    Returns:
        safety_car_probability: P(at least one SC)
        expected_safety_cars: E[number of SCs]
        vsc_probability: P(VSC only, no full SC)
    """
```

### Circuit Safety Car Probabilities (Historical)

| Circuit | SC Probability | Expected SCs | VSC Probability |
|---------|----------------|--------------|-----------------|
| Monaco | 0.65 | 0.9 | 0.35 |
| Baku | 0.72 | 1.1 | 0.25 |
| Singapore | 0.58 | 0.8 | 0.30 |
| Silverstone | 0.35 | 0.4 | 0.20 |
| Monza | 0.42 | 0.5 | 0.22 |

### Monte Carlo Safety Car Integration

```python
# Simulate safety car occurrences
sc_occurred = rng.random(n_simulations) < safety_car_probability

# Adjust pit stop penalty when SC occurs
adjusted_penalty = np.where(
    sc_occurred & (extra_stops > 0),
    base_penalty * 0.3,  # SC reduces penalty by 70%
    base_penalty
)
```

### Impact on Strategy Selection

- High SC probability favors multi-stop strategies (free pit stops during SC)
- SC benefit is added to Monte Carlo output for each strategy
- Tactical recommendation generated when SC probability > 50%

## 3. Weather Integration

### Weather Impact Calculation

```python
def get_weather_impact(
    track_temp: float = None,
    air_temp: float = None,
    humidity: float = None,
    rain_probability: float = 0.0,
    is_wet_race: bool = False,
) -> Dict:
```

### Temperature Effects

| Track Temp | Degradation Multiplier | Adjustment |
|------------|------------------------|------------|
| > 50°C | 1.4x | high_degradation |
| 40-50°C | 1.2x | moderate |
| 25-40°C | 1.0x | normal |
| < 25°C | 0.8x | low_degradation |

### Rain Conditions

| Condition | Weather Factor | Recommended Compounds |
|-----------|----------------|----------------------|
| Dry | 1.0 | SOFT, MEDIUM, HARD |
| Light rain risk (30-70%) | 1.3 | INTERMEDIATE, MEDIUM, SOFT |
| Heavy rain (>70%) | 1.5 | WET, INTERMEDIATE |

### Weather Factor Application

The weather factor increases uncertainty in Monte Carlo simulation:
```python
base_std = predicted_position_std * weather_factor
deg_noise = rng.normal(0, degradation_std * weather_factor, n_simulations)
```

## 4. Competitor Modeling

### Undercut/Overcut Analysis

```python
def calculate_undercut_overcut_windows(
    driver_position: int,
    lap_delta_to_car_ahead: float,
    lap_delta_to_car_behind: float,
    pit_loss_sec: float = 22.0,
    tyre_advantage_laps: int = 3,
) -> Dict:
```

### Undercut Calculation

The undercut works by pitting earlier to gain time on fresh tyres:

```
fresh_tyre_gain_per_lap = 0.7 seconds
undercut_gain = tyre_advantage_laps × fresh_tyre_gain_per_lap
undercut_viable = undercut_gain > gap_to_car_ahead AND gap < 3.0s
undercut_confidence = min(1.0, undercut_gain / gap_to_car_ahead)
```

### Overcut Calculation

The overcut works by staying out while competitors pit:

```
overcut_viable = gap_to_car_behind > 2.5s AND position > 1
overcut_confidence = min(1.0, gap_to_car_behind / 5.0)
```

### Traffic Analysis

```python
traffic_analysis = {
    "in_traffic": gap_to_car_ahead < 1.5,
    "clear_air_available": gap_to_car_ahead > 3.0,
}
```

### Decision Matrix

| Gap Ahead | Gap Behind | Recommendation |
|-----------|------------|----------------|
| < 1.5s | any | Consider undercut (pit early) |
| > 3.0s | > 2.5s | Consider overcut (stay out) |
| 1.5-3.0s | < 2.5s | Standard pit window |
| any | < 1.5s | Beware undercut from behind |

## 5. Monte Carlo Simulation (Enhanced)

### Simulation Model

```
P_final = P_predicted + ε_base + ε_degradation + ε_traffic + penalty_pit + ε_strategy

where:
- P_predicted ~ N(mean, std × weather_factor)
- ε_degradation ~ N(0, σ_deg × weather_factor)
- ε_traffic ~ N(0, σ_traffic)
- penalty_pit adjusted by safety_car_occurred
- ε_strategy ~ N(0, extra_stops × 0.5)
```

### Enhanced Output

```python
{
    "strategy_id": 0,
    "label": "one-stop",
    "expected_position": 5.3,
    "std_position": 2.1,
    "typical_stops": 1,
    "best_case": 1.5,        # Minimum in simulation
    "worst_case": 12.3,      # Maximum in simulation
    "percentile_25": 4.1,    # 25th percentile
    "percentile_75": 6.8,    # 75th percentile
    "sc_benefit": 0.8,       # Position benefit from SC
    "rank": 1
}
```

## 6. Tactical Recommendations

The system generates actionable tactical recommendations based on analysis:

```python
tactical_recommendations = [
    {
        "type": "undercut_opportunity",
        "action": "Consider pitting 1-2 laps earlier than planned",
        "confidence": 0.75,
    },
    {
        "type": "safety_car_likely",
        "action": "Prepare for opportunistic pit stop during SC",
        "confidence": 0.65,
    },
    {
        "type": "high_degradation",
        "action": "Favor harder compound choices",
        "confidence": 0.80,
    },
    {
        "type": "rain_risk",
        "action": "Keep intermediates ready, monitor conditions",
        "confidence": 0.40,
    },
]
```

## API Integration

### Endpoint: `/strategy`

#### Full Request Example
```
GET /strategy?
    predicted_position_mean=5.0&
    predicted_position_std=2.0&
    circuit_id=monaco&
    race_laps=78&
    track_temp=45&
    air_temp=28&
    humidity=60&
    rain_probability=0.2&
    is_wet_race=false&
    gap_to_car_ahead=1.5&
    gap_to_car_behind=3.0&
    pit_loss_sec=22&
    n_simulations=2000
```

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| predicted_position_mean | float | required | 1-20 | Expected finish position |
| predicted_position_std | float | 2.0 | 0.1-10 | Position uncertainty |
| circuit_id | string | null | - | Circuit for SC probability |
| race_laps | int | 56 | 30-80 | Total race laps |
| track_temp | float | null | 10-70 | Track temperature (°C) |
| air_temp | float | null | 5-50 | Air temperature (°C) |
| humidity | float | null | 0-100 | Humidity (%) |
| rain_probability | float | 0.0 | 0-1 | Rain probability |
| is_wet_race | bool | false | - | Wet race flag |
| gap_to_car_ahead | float | 1.0 | 0-60 | Gap ahead (seconds) |
| gap_to_car_behind | float | 1.0 | 0-60 | Gap behind (seconds) |
| pit_loss_sec | float | 22.0 | 15-35 | Pit stop time loss |
| n_simulations | int | 2000 | 100-10000 | MC simulation count |

#### Response Structure

```json
{
  "predicted_position_mean": 5.0,
  "predicted_position_std": 2.0,
  "best_strategy": {
    "strategy_id": 0,
    "label": "one-stop",
    "expected_position": 5.3,
    "std_position": 2.1,
    "typical_stops": 1.0,
    "best_case": 1.5,
    "worst_case": 12.3,
    "percentile_25": 4.1,
    "percentile_75": 6.8,
    "sc_benefit": 0.0,
    "rank": 1
  },
  "strategy_ranking": [...],
  "strategy_summary": [...],
  "safety_car_analysis": {
    "safety_car_probability": 0.65,
    "expected_safety_cars": 0.9,
    "vsc_probability": 0.35,
    "historical_races_analyzed": 10,
    "circuit_id": "monaco"
  },
  "weather_impact": {
    "weather_factor": 1.2,
    "degradation_multiplier": 1.2,
    "recommended_compounds": ["MEDIUM", "HARD", "SOFT"],
    "strategy_adjustments": [],
    "conditions": {
      "track_temp": 45,
      "air_temp": 28,
      "humidity": 60,
      "rain_probability": 0.2,
      "is_wet_race": false
    }
  },
  "competitor_analysis": {
    "undercut": {
      "viable": true,
      "confidence": 0.75,
      "potential_gain_positions": 1,
      "recommended_action": "pit_early"
    },
    "overcut": {
      "viable": true,
      "confidence": 0.60,
      "potential_gain_positions": 1,
      "recommended_action": "stay_out"
    },
    "traffic_analysis": {
      "in_traffic": true,
      "clear_air_available": false,
      "gap_to_car_ahead": 1.5,
      "gap_to_car_behind": 3.0
    },
    "position": 5
  },
  "compound_strategies": [
    {
      "strategy_id": 0,
      "label": "S-M (1-stop)",
      "stints": [
        {"stint": 1, "compound": "SOFT", "laps": 15},
        {"stint": 2, "compound": "MEDIUM", "laps": 41}
      ],
      "total_stops": 1,
      "compound_sequence": ["SOFT", "MEDIUM"],
      "pace_score": 0.65,
      "risk_score": 0.15,
      "overall_score": 0.41
    },
    ...
  ],
  "tactical_recommendations": [
    {
      "type": "undercut_opportunity",
      "action": "Consider pitting 1-2 laps earlier than planned",
      "confidence": 0.75
    },
    {
      "type": "safety_car_likely",
      "action": "Prepare for opportunistic pit stop during SC",
      "confidence": 0.65
    }
  ],
  "race_parameters": {
    "race_laps": 78,
    "pit_loss_sec": 22.0,
    "circuit_id": "monaco"
  }
}
```

## Usage Examples

### Example 1: Front-Runner at Monaco

```python
# Driver expected to finish P2 with low uncertainty at Monaco
result = recommend_strategy(
    predicted_position_mean=2.0,
    predicted_position_std=1.5,
    circuit_id="monaco",
    race_laps=78,
    track_temp=35,
    gap_to_car_ahead=2.5,
    gap_to_car_behind=8.0,
)
# Result: One-stop recommended (minimize time in pits)
# SC probability ~65% - prepare for opportunistic stop
# Overcut viable due to large gap behind
```

### Example 2: Midfield Battle in Rain

```python
# Driver in tight midfield battle with rain expected
result = recommend_strategy(
    predicted_position_mean=10.0,
    predicted_position_std=3.0,
    circuit_id="silverstone",
    race_laps=52,
    track_temp=25,
    rain_probability=0.6,
    gap_to_car_ahead=1.0,
    gap_to_car_behind=1.2,
)
# Result: Weather factor increased to 1.3
# Intermediates recommended as standby
# Undercut viable (small gap ahead)
```

### Example 3: Recovery Drive with High Degradation

```python
# Driver starting from back, high track temp
result = recommend_strategy(
    predicted_position_mean=15.0,
    predicted_position_std=4.0,
    circuit_id="bahrain",
    race_laps=57,
    track_temp=55,
    gap_to_car_ahead=5.0,
    gap_to_car_behind=2.0,
)
# Result: High degradation multiplier (1.4x)
# Two-stop may be competitive with SC possibility
# Harder compounds recommended
```

## Algorithm Complexity

| Component | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Cluster Strategies | O(n × k × i) | O(n) |
| Safety Car Prob | O(r) | O(r) |
| Weather Impact | O(1) | O(1) |
| Undercut/Overcut | O(1) | O(1) |
| Compound Gen | O(c³) | O(c³) |
| Monte Carlo | O(s × n) | O(s) |

Where: n = data points, k = clusters, i = iterations, r = races, c = compounds, s = simulations

## Limitations

### Current Limitations
1. **Static Degradation Model**: Compound characteristics are constants, not dynamic
2. **No Tyre Compound Availability**: Assumes all compounds available
3. **Simplified Traffic Model**: Binary in-traffic / clear-air assessment
4. **No Fuel Load Consideration**: Weight effect not modeled
5. **Historical SC Data Only**: No predictive SC model

### Assumptions
1. Pit stop execution is consistent (~22 seconds loss)
2. Undercut advantage is ~0.7s per lap on fresh tyres
3. Degradation is linear (not exponential cliff)
4. Weather conditions are stable during race
5. All strategies are equally executable by team

## Future Enhancements

1. **Dynamic Degradation Curves**
   - Model compound-specific degradation cliffs
   - Include fuel load effect on lap times

2. **Real-Time Integration**
   - Live timing data integration
   - Dynamic gap updates during race

3. **Predictive Safety Car**
   - Weather-based SC probability
   - First lap incident modeling

4. **Tyre Availability**
   - Track available compound sets per race
   - New vs used tyre modeling

5. **Team Performance**
   - Team-specific pit stop times
   - Historical reliability factors

## References

1. F1 Pit Stop Analysis: https://f1metrics.wordpress.com/
2. Monte Carlo Methods: https://en.wikipedia.org/wiki/Monte_Carlo_method
3. KMeans Clustering: https://scikit-learn.org/stable/modules/clustering.html
4. Pirelli Tyre Compounds: https://press.pirelli.com/f1-tyres/
