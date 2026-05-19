from backend.agents.policy import compact_scores, normalize, ranked, softmax_confidence


CONFLICT_RULES = [
    {
        "type": "route_perimeter",
        "ambulance_weight": 0.58,
        "police_weight": 0.42,
        "trigger": lambda ambulance, traffic, police, hospital: police["perimeter_radius_m"] >= 100,
        "description": lambda ambulance, traffic, police, hospital: (
            f"Ambulance route via {ambulance['route']} intersects "
            f"police perimeter ({police['perimeter_radius_m']}m zone on "
            f"{police['perimeter_road']})"
        ),
    },
    {
        "type": "hospital_capacity",
        "ambulance_weight": 0.36,
        "police_weight": 0.12,
        "hospital_weight": 0.52,
        "trigger": lambda ambulance, traffic, police, hospital: hospital["recommended_beds"]
        < hospital["beds_reserved"],
        "description": lambda ambulance, traffic, police, hospital: (
            f"Recommended hospital has {hospital['recommended_beds']} beds but "
            f"{hospital['beds_reserved']} were requested"
        ),
    },
]


def priority_bids(ambulance: dict, police: dict, hospital: dict, rule: dict) -> dict:
    bids = {
        "Ambulance": (
            normalize(ambulance["eta_minutes"], 35, 1) * 4.0
            + normalize(ambulance["ambulances_dispatched"], 1, 3) * 2.5
            + ambulance["confidence"] * 2.0
            + rule.get("ambulance_weight", 0.0) * 2.0
        ),
        "Police": (
            normalize(police["perimeter_radius_m"], 50, 220) * 2.7
            + normalize(police["units_dispatched"], 2, 5) * 2.0
            + police["confidence"] * 1.8
            + rule.get("police_weight", 0.0) * 2.0
        ),
    }
    if "hospital_weight" in rule:
        bids["Hospital"] = (
            normalize(hospital["beds_reserved"], 1, 8) * 2.4
            + hospital["confidence"] * 2.2
            + rule["hospital_weight"] * 2.0
        )
    return {agent: round(score, 2) for agent, score in bids.items()}


def coordinator(ambulance: dict, traffic: dict, police: dict, hospital: dict) -> dict:
    conflicts = []
    scored_conflicts = []
    resolution = None

    for rule in CONFLICT_RULES:
        if rule["trigger"](ambulance, traffic, police, hospital):
            bids = priority_bids(ambulance, police, hospital, rule)
            conflicts.append(
                {
                    "type": rule["type"],
                    "description": rule["description"](ambulance, traffic, police, hospital),
                    "agents": list(bids.keys()),
                    "bids": bids,
                }
            )

    if conflicts:
        scored_conflicts = [
            {**item, "score": max(item["bids"].values()) + len(item["agents"]) * 0.1}
            for item in conflicts
        ]
        conflict = ranked(scored_conflicts)[0]
        winner = max(conflict["bids"], key=conflict["bids"].get)
        loser = min(conflict["bids"], key=conflict["bids"].get)
        updated_eta = ambulance["eta_minutes"] + 2
        action = f"{loser} agent yields corridor. Ambulance re-routed via {ambulance['alt_route']}."
        if winner == "Hospital":
            action = f"{loser} agent yields. Patient intake stays with {hospital['recommended']}."
        resolution = {
            "conflict": conflict["type"],
            "winner": winner,
            "loser": loser,
            "action": action,
            "eta_updated": updated_eta,
            "confidence": softmax_confidence(list(conflict["bids"].values())),
        }
        ambulance["eta_minutes"] = updated_eta
        ambulance["route"] = ambulance["alt_route"]

    return {
        "conflicts_detected": len(conflicts),
        "conflicts": [
            {key: value for key, value in conflict.items() if key != "score"}
            for conflict in conflicts
        ],
        "resolution": resolution,
        "final_eta": ambulance["eta_minutes"],
        "final_hospital": hospital["recommended"],
        "final_route": ambulance["route"],
        "negotiation_candidates": compact_scores(scored_conflicts, ("type", "agents")) if scored_conflicts else [],
        "resource_summary": {
            "ambulances": ambulance["ambulances_dispatched"],
            "paramedic_teams": ambulance["paramedic_teams"],
            "police_units": police["units_dispatched"],
            "signals_overridden": traffic["junctions_overridden"],
            "beds_reserved": hospital["beds_reserved"],
        },
        "status": "DISPATCHED" if not conflicts or resolution else "PENDING",
    }
