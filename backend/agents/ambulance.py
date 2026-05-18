import datetime
import numpy as np
import pandas as pd

from agents.policy import compact_scores, inverse_normalize, ranked, softmax_confidence, stable_rng
from data import AMBULANCE_BASES, LOCATION_COORDS, ROUTES_BY_LOCATION
from routing import batch_ambulance_distances

TRIAGE_BANDS = [
    {"label": "P1 immediate", "target": 1.0},
    {"label": "P2 urgent",    "target": 0.65},
    {"label": "P3 stable",    "target": 0.3},
]

ROUTE_OPTIONS = [
    {"name": "Brigade Road",        "traffic_fit": 0.72, "signal_count": 3, "critical_access": 0.88},
    {"name": "Residency Road",      "traffic_fit": 0.86, "signal_count": 2, "critical_access": 0.66},
    {"name": "Hosur Road corridor", "traffic_fit": 0.93, "signal_count": 4, "critical_access": 0.92},
    {"name": "direct route",        "traffic_fit": 0.68, "signal_count": 1, "critical_access": 0.58},
]

# ── BENGALURU AMBULANCE SPEED CONSTANTS ─────────────────────────────────────
# Realistic effective ambulance speeds (with sirens, signal pre-emption active)
# Based on BBMP EMS data: avg response 14–22 min for 3–8km in city.
# Peak hours ambulances rarely exceed 25 km/h effective in dense corridors.
_AMBULANCE_SPEED = {"peak": 20, "normal": 28, "night": 40}

# Bengaluru-specific constants
DISPATCH_OVERHEAD_MIN   = 4    # call received → crew boards → exits bay (fixed overhead)
MOBILIZATION_MIN        = 5    # minimum ETA even if base is very nearby
BLR_CONGESTION_BASE     = 1.20 # Bengaluru baseline congestion factor always applied
SIGNAL_DELAY_PER_KM     = 0.65 # ~40s per signal, ~1 signal per km on emergency routes
MIN_ROAD_DISTANCE_KM    = 1.0  # realistic minimum — base is never co-located with incident

def _ambulance_speed_kmph() -> int:
    h = datetime.datetime.now().hour
    if h in range(8, 11) or h in range(17, 21):
        return _AMBULANCE_SPEED["peak"]
    if h in range(0, 6) or h == 23:
        return _AMBULANCE_SPEED["night"]
    return _AMBULANCE_SPEED["normal"]

def _physics_eta(distance_km: float, traffic: int, weather: int) -> float:
    """
    Physics-based ETA for Bengaluru conditions.
    Used only when ORS is unavailable. Accounts for:
    - Time-of-day speed (peak / normal / night)
    - Traffic congestion degradation
    - Weather degradation (rain)
    - Signal delays per km
    - Dispatch overhead
    Minimum realistic ETA = MOBILIZATION_MIN minutes.
    """
    distance_km = max(MIN_ROAD_DISTANCE_KM, distance_km)
    speed = _ambulance_speed_kmph()

    # Traffic degrades speed: level 0=no effect, level 3=−42%
    traffic_factor  = 1.0 - (traffic * 0.14)
    # Weather: rain degrades −12%
    weather_factor  = 1.0 - (weather * 0.06)
    # Bengaluru base congestion always applied
    effective_speed = max(12, speed * BLR_CONGESTION_BASE * traffic_factor * weather_factor)

    # Drive time + signal delays
    drive_min         = (distance_km / effective_speed) * 60
    signal_delay_min  = distance_km * SIGNAL_DELAY_PER_KM
    eta               = drive_min + signal_delay_min + DISPATCH_OVERHEAD_MIN

    return round(max(MOBILIZATION_MIN, eta), 1)


def select_triage_priority(severity, injured):
    acuity  = min(1.0, (severity * 0.16) + (injured * 0.045))
    choices = [{**band, "score": 1.0 - abs(acuity - band["target"])} for band in TRIAGE_BANDS]
    return ranked(choices)[0]["label"]


def ambulance_agent(parsed, eta_model, model_rmse):
    severity = parsed["severity"]
    location = parsed["location"]
    injured  = parsed["injured"]
    traffic  = 3 if parsed.get("traffic_hint") == "heavy" else min(severity - 1, 3)
    weather  = 2 if parsed.get("weather_hint") == "rain" else 0
    rng      = stable_rng(parsed["raw"], "ambulance")

    acc_coords = LOCATION_COORDS.get(location, {"lat": 12.9716, "lon": 77.5946})
    road_dists = batch_ambulance_distances(
        acc_coords["lat"], acc_coords["lon"], AMBULANCE_BASES
    )

    unit_scores = []
    for unit in AMBULANCE_BASES:
        rd          = road_dists.get(unit["id"], {})
        road_dist   = rd.get("distance_km", 8.0)
        road_dur    = rd.get("duration_min", None)
        dist_source = rd.get("source", "haversine_fallback")

        # Small jitter so identical-distance bases don't always tie
        distance = round(max(MIN_ROAD_DISTANCE_KM, road_dist + rng.normal(0, 0.15)), 1)

        # ── ETA CALCULATION ──────────────────────────────────────────────────
        # Priority:
        #   1. ORS duration (real road graph time) + Bengaluru congestion adjustments
        #   2. Physics-based fallback using distance
        #
        # ORS returns FREE-FLOW time. For Bengaluru, actual travel is 1.8–2.5x
        # free-flow during peak. We apply:
        #   - BLR_CONGESTION_BASE (always-on Bengaluru baseline: 1.20×)
        #   - traffic_mult (incident-specific congestion: up to +42%)
        #   - weather_mult (rain: +12%)
        #   - DISPATCH_OVERHEAD_MIN (fixed 4 min crew mobilization)
        if road_dur is not None and road_dur > 0:
            traffic_mult = BLR_CONGESTION_BASE * (1.0 + (traffic * 0.14))  # peak heavy = 1.20×1.42 = 1.70×
            weather_mult = 1.0 + (weather * 0.06)
            drive_eta    = round(road_dur * traffic_mult * weather_mult, 1)
            eta          = round(max(MOBILIZATION_MIN, drive_eta + DISPATCH_OVERHEAD_MIN), 1)
        else:
            eta = _physics_eta(distance, traffic, weather)

        load  = min(0.95, max(0.05, unit["load"] + rng.normal(0, 0.05)))
        score = (
            inverse_normalize(eta, 4, 45) * 0.52
            + unit["capability"]            * 0.26
            + inverse_normalize(load, 0, 1) * 0.14
            + min(1.0, severity / 5)        * 0.08
        )
        unit_scores.append({
            **unit,
            "load":        round(load, 2),
            "distance_km": distance,
            "eta_minutes": eta,
            "dist_source": dist_source,
            "score":       score,
        })

    units_ranked  = ranked(unit_scores)
    selected_unit = units_ranked[0]
    distance      = selected_unit["distance_km"]
    eta           = round(selected_unit["eta_minutes"])

    # ── SANITY CHECKS ────────────────────────────────────────────────────────
    # Floor 1: physics minimum — 25 km/h is the absolute best possible speed
    #          in Bengaluru with sirens + signal pre-emption. No ambulance
    #          can legally or physically beat this.
    physics_floor = round((distance / 25) * 60 + DISPATCH_OVERHEAD_MIN)
    # Floor 2: absolute minimum — no dispatch is ever under MOBILIZATION_MIN
    eta = max(eta, physics_floor, MOBILIZATION_MIN)

    route_options = ROUTES_BY_LOCATION.get(location, ROUTE_OPTIONS)
    route_scores  = []
    for option in route_options:
        score = (
            option["traffic_fit"]       * 0.45
            + option["critical_access"] * min(1.0, severity / 5) * 0.30
            + inverse_normalize(option["signal_count"], 1, 5)    * 0.12
            + inverse_normalize(traffic, 0, 3)                   * 0.13
            - parsed.get("access_complexity", 0.5)               * 0.04
        )
        route_scores.append({**option, "score": score})

    routes_ranked   = ranked(route_scores)
    route           = routes_ranked[0]["name"]
    alt_route       = routes_ranked[1]["name"] if len(routes_ranked) > 1 else route
    ambulances      = max(1, min(3, (injured + 1) // 2))
    paramedic_teams = max(ambulances, round((severity + injured) / 3))
    priority        = select_triage_priority(severity, injured)
    confidence      = softmax_confidence([u["score"] for u in units_ranked])

    return {
        "agent":                 "Ambulance",
        "model":                 "ORS Road Distance + Physics ETA",
        "selected_unit":         selected_unit["id"],
        "unit_base":             selected_unit["name"],
        "unit_lat":              selected_unit["lat"],
        "unit_lon":              selected_unit["lon"],
        "distance_km":           distance,
        "eta_minutes":           eta,
        "dist_source":           selected_unit["dist_source"],
        "route":                 route,
        "alt_route":             alt_route,
        "ambulances_dispatched": ambulances,
        "paramedic_teams":       paramedic_teams,
        "triage_priority":       priority,
        "signal_override":       True,
        "confidence":            confidence,
        "unit_candidates":       compact_scores(
            units_ranked, ("id", "name", "distance_km", "eta_minutes", "load", "dist_source")
        ),
        "route_candidates":      compact_scores(routes_ranked, ("name", "signal_count")),
        "reasoning": (
            f"Selected {selected_unit['id']} from {selected_unit['name']} "
            f"({distance}km road distance, ETA {eta} min incl. Bengaluru congestion/"
            f"traffic/weather/signals + {DISPATCH_OVERHEAD_MIN}min dispatch overhead, "
            f"source={selected_unit['dist_source']}). "
            f"Dispatching {ambulances} ambulance(s), {paramedic_teams} paramedic team(s). "
            f"Route: {route}; fallback: {alt_route}."
        ),
    }
