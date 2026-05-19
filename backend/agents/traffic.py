from backend.agents.policy import compact_scores, inverse_normalize, normalize, ranked, softmax_confidence
from backend.ml.ml_models import traffic_feature_frame


CONGESTION_LABELS = ["LOW", "MODERATE", "HIGH", "SEVERE"]

TRAFFIC_MODES = [
    {"mode": "monitor only", "delay_reduction": 0.15, "disruption": 0.05, "control_strength": 0.2, "target_pressure": 0.2},
    {"mode": "adaptive green wave", "delay_reduction": 0.45, "disruption": 0.25, "control_strength": 0.55, "target_pressure": 0.45},
    {"mode": "priority corridor locked", "delay_reduction": 0.72, "disruption": 0.55, "control_strength": 0.82, "target_pressure": 0.72},
    {"mode": "full arterial lockdown", "delay_reduction": 0.88, "disruption": 0.82, "control_strength": 1.0, "target_pressure": 0.95},
]

DIVERSION_OPTIONS = [
    {"road": "Old Airport Road", "capacity": 0.78, "distance_penalty": 0.24, "zones": ["east", "central"]},
    {"road": "Richmond Circle", "capacity": 0.7, "distance_penalty": 0.18, "zones": ["central", "south"]},
    {"road": "local slip roads", "capacity": 0.46, "distance_penalty": 0.08, "zones": ["central", "east", "south", "north"]},
    {"road": "Outer Ring Road feeder", "capacity": 0.86, "distance_penalty": 0.42, "zones": ["east", "south", "north"]},
    {"road": "Bellary Road service lane", "capacity": 0.68, "distance_penalty": 0.2, "zones": ["north"]},
    {"road": "BTM internal diversion", "capacity": 0.58, "distance_penalty": 0.16, "zones": ["south"]},
]


def traffic_agent(parsed: dict, traffic_models: dict) -> dict:
    severity = parsed["severity"]
    location = parsed["location"]
    features = traffic_feature_frame(parsed)
    delay = round(float(traffic_models["delay"].predict(features)[0]))
    index = int(traffic_models["class"].predict(features)[0])
    class_probs = traffic_models["class"].predict_proba(features)[0]
    ml_confidence = round(float(max(class_probs)), 2)
    traffic_pressure = min(1.0, max(0.0, delay / 28))
    congestion_level = CONGESTION_LABELS[index]
    junctions = max(2, round(2 + severity + traffic_pressure * 2))

    mode_scores = []
    for mode in TRAFFIC_MODES:
        suitability = 1.0 - abs(traffic_pressure - mode["target_pressure"])
        score = (
            mode["delay_reduction"] * traffic_pressure * 0.34
            + mode["control_strength"] * normalize(severity, 1, 5) * 0.26
            + inverse_normalize(mode["disruption"], 0, 1) * 0.15
            + suitability * 0.25
        )
        mode_scores.append({**mode, "score": score})

    modes_ranked = ranked(mode_scores)
    selected_mode = modes_ranked[0]

    diversion_scores = []
    for option in DIVERSION_OPTIONS:
        zone_fit = 1.0 if parsed.get("location_zone") in option["zones"] else 0.45
        score = (
            option["capacity"] * traffic_pressure * 0.62
            + inverse_normalize(option["distance_penalty"], 0, 0.5) * 0.25
            + normalize(severity, 1, 5) * 0.08
            + zone_fit * 0.18
        )
        diversion_scores.append({**option, "score": score})

    diversions_ranked = ranked(diversion_scores)
    diversion_roads = [option["road"] for option in diversions_ranked[:2]]
    confidence = softmax_confidence([mode["score"] for mode in modes_ranked])

    return {
        "agent": "Traffic",
        "model": "RandomForest Traffic Predictor + Signal Policy Ranker",
        "location": location,
        "congestion_level": congestion_level,
        "predicted_delay_min": delay,
        "junctions_overridden": junctions,
        "corridor_status": selected_mode["mode"],
        "diversion_roads": diversion_roads,
        "signal_override": True,
        "traffic_pressure": round(traffic_pressure, 2),
        "confidence": round((confidence + ml_confidence) / 2, 2),
        "ml_confidence": ml_confidence,
        "mode_candidates": compact_scores(modes_ranked, ("mode", "delay_reduction", "disruption")),
        "diversion_candidates": compact_scores(diversions_ranked, ("road", "capacity", "zones")),
        "reasoning": (
            f"RandomForest traffic model forecasts {congestion_level} congestion on {location}. "
            f"Predicted delay: {delay} min (class confidence={ml_confidence}). "
            f"Overriding {junctions} traffic signals on emergency corridor. "
            f"Policy ranker selected mode: {selected_mode['mode']}."
        ),
    }
