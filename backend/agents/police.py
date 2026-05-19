from backend.agents.policy import compact_scores, normalize, ranked, softmax_confidence


DEPLOYMENT_PLANS = [
    {
        "name": "lane control",
        "units": 2,
        "radius": 60,
        "investigation": 0.25,
        "crowd_control": 0.35,
        "disruption": 0.15,
    },
    {
        "name": "traffic diversion + crowd control",
        "units": 3,
        "radius": 100,
        "investigation": 0.45,
        "crowd_control": 0.62,
        "disruption": 0.34,
    },
    {
        "name": "major accident investigation + crowd control",
        "units": 4,
        "radius": 150,
        "investigation": 0.82,
        "crowd_control": 0.78,
        "disruption": 0.58,
    },
    {
        "name": "mass casualty perimeter lockdown",
        "units": 5,
        "radius": 220,
        "investigation": 0.95,
        "crowd_control": 0.92,
        "disruption": 0.86,
    },
]

PERIMETER_ROADS = [
    {"road": "east lane", "access_conflict": 0.2, "containment": 0.42},
    {"road": "south access road", "access_conflict": 0.38, "containment": 0.75},
    {"road": "service road shoulder", "access_conflict": 0.12, "containment": 0.58},
]

CROWD_RISK_BANDS = [
    {"label": "low", "target": 0.25},
    {"label": "medium", "target": 0.55},
    {"label": "high", "target": 0.85},
]


def police_agent(parsed: dict) -> dict:
    severity = parsed["severity"]
    location = parsed["location"]
    injured = parsed["injured"]
    risk_index = min(1.0, severity * 0.16 + injured * 0.055)

    plan_scores = []
    for plan in DEPLOYMENT_PLANS:
        score = (
            plan["investigation"] * normalize(severity, 1, 5) * 0.36
            + plan["crowd_control"] * risk_index * 0.34
            + normalize(plan["units"], 2, 5) * normalize(injured, 1, 8) * 0.18
            + (1.0 - plan["disruption"]) * 0.12
        )
        plan_scores.append({**plan, "score": score})

    plans_ranked = ranked(plan_scores)
    selected_plan = plans_ranked[0]

    road_scores = []
    for road in PERIMETER_ROADS:
        score = (
            road["containment"] * risk_index * 0.66
            + (1.0 - road["access_conflict"]) * 0.24
            + normalize(severity, 1, 5) * 0.1
        )
        road_scores.append({**road, "score": score})

    roads_ranked = ranked(road_scores)
    crowd_risk = ranked(
        [
            {**band, "score": 1.0 - abs(risk_index - band["target"])}
            for band in CROWD_RISK_BANDS
        ]
    )[0]["label"]
    confidence = softmax_confidence([plan["score"] for plan in plans_ranked])

    return {
        "agent": "Police",
        "model": "TF-IDF Logistic Severity Classifier + Deployment Ranker",
        "severity_classified": severity,
        "rule_severity": parsed.get("rule_severity", severity),
        "ml_severity": parsed.get("ml_severity", severity),
        "ml_severity_confidence": parsed.get("ml_severity_confidence", 0.0),
        "units_dispatched": selected_plan["units"],
        "perimeter_radius_m": selected_plan["radius"],
        "perimeter_road": roads_ranked[0]["road"],
        "crowd_risk": crowd_risk,
        "protocol": selected_plan["name"],
        "risk_index": round(risk_index, 2),
        "confidence": confidence,
        "deployment_candidates": compact_scores(plans_ranked, ("name", "units", "radius")),
        "perimeter_candidates": compact_scores(roads_ranked, ("road", "containment")),
        "reasoning": (
            f"TF-IDF LogisticRegression classifier predicted severity "
            f"{parsed.get('ml_severity', severity)}/5 "
            f"(confidence={parsed.get('ml_severity_confidence', 0.0)}). "
            f"Blended operational severity = {severity}/5. "
            f"Deployment policy selected {selected_plan['name']} for {location}. "
            f"Dispatching {selected_plan['units']} units and establishing "
            f"{selected_plan['radius']}m perimeter on {roads_ranked[0]['road']}. "
            f"Crowd risk: {crowd_risk}."
        ),
    }
