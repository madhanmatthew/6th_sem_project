import numpy as np

from agents.policy import compact_scores, inverse_normalize, normalize, ranked, softmax_confidence, stable_rng
from data import HOSPITALS, LOCATION_COORDS
from ml_models import hospital_feature_frame
from routing import batch_hospital_distances


def hospital_agent(parsed, hospital_model):
    severity = parsed["severity"]
    injured  = parsed["injured"]
    location = parsed["location"]
    rng      = stable_rng(parsed["raw"], "hospital")
    hospitals = []

    # Single ORS API call for ALL hospitals — real road distances
    acc_coords = LOCATION_COORDS.get(location, {"lat": 12.9716, "lon": 77.5946})
    road_dists = batch_hospital_distances(
        acc_coords["lat"], acc_coords["lon"], HOSPITALS
    )

    for hospital in HOSPITALS:
        # Realistic occupancy: 55-85% typical for Bengaluru hospitals
        base_occupancy = 0.60 + (parsed.get("location_risk", 0.5) * 0.15)
        occupancy_rate = min(0.90, base_occupancy + rng.uniform(-0.05, 0.10))
        free_beds = max(1, int(hospital["capacity"] * (1.0 - occupancy_rate)))
        load_ratio = round(1.0 - (free_beds / hospital["capacity"]), 2)

        # Real road distance from ORS (falls back to haversine×1.4 if no key)
        rd          = road_dists.get(hospital["name"], {})
        distance    = rd.get("distance_km", 9.0)
        drive_time  = rd.get("duration_min", None)
        dist_source = rd.get("source", "haversine_fallback")

        trauma_required = severity >= 3
        capacity_fit    = min(1.0, free_beds / max(1, injured))

        # Specialty bonus
        specialty_bonus = 0.0
        if parsed.get("incident_type") == "traffic collision" and hospital.get("accident_victims"):
            specialty_bonus = 0.08
        if severity >= 4 and hospital.get("polytrauma"):
            specialty_bonus += 0.06
        if severity >= 4 and hospital.get("neurosurgery"):
            specialty_bonus += 0.04

        # Policy score: distance (35%) + beds (25%) + trauma (20%) + load (12%) + specialty (8%)
        dist_score   = inverse_normalize(distance, 1, 25)   # 25km max road range
        bed_score    = min(1.0, free_beds / max(1, injured * 2))
        trauma_score = 1.0 if (hospital["trauma"] or not trauma_required) else 0.15
        load_score   = 1.0 - load_ratio
        policy_score = (
            dist_score    * 0.35
            + bed_score   * 0.25
            + trauma_score * 0.20
            + load_score  * 0.12
            + specialty_bonus * 0.08
        )

        ml_score = float(hospital_model.predict(
            hospital_feature_frame(severity, injured, int(free_beds),
                                   hospital["trauma"], distance, load_ratio)
        )[0])

        # Policy drives (80%), ML tiebreaker (20%)
        score = (policy_score * 0.80) + (ml_score * 0.20)

        # Hard penalties
        if free_beds < injured:
            score -= 0.25
        if trauma_required and not hospital["trauma"]:
            score -= 0.30

        hospitals.append({
            **hospital,
            "free_beds":    int(free_beds),
            "distance_km":  distance,
            "drive_min":    drive_time,
            "dist_source":  dist_source,
            "load_ratio":   load_ratio,
            "capacity_fit": round(capacity_fit, 2),
            "ml_score":     round(ml_score, 3),
            "policy_score": round(policy_score, 3),
            "score":        score,
        })

    scored   = ranked(hospitals)
    chosen   = scored[0]
    fallback = scored[1]
    beds_reserved   = min(chosen["free_beds"], max(1, injured))
    trauma_required = severity >= 3
    confidence      = softmax_confidence([h["score"] for h in scored])

    # ── ETA to hospital ───────────────────────────────────────────────────────
    # Physics floor: Bengaluru peak traffic — 20 km/h effective + signal delays
    # (0.5 min/km for ~30s per signal, 1 signal/km on city roads)
    # ORS free-flow time is multiplied by Bengaluru congestion factor (1.20×)
    # before comparing with physics floor.
    _physics_floor = round((chosen["distance_km"] / 20) * 60 + chosen["distance_km"] * 0.5)
    _ors_eta       = round(chosen["drive_min"] * 1.20, 1) if chosen["drive_min"] else None
    eta_to_hospital = max(_physics_floor, _ors_eta) if _ors_eta else _physics_floor
    # Minimum 5 min — no hospital arrival is credible under this in Bengaluru city
    eta_to_hospital = max(5, eta_to_hospital)

    return {
        "agent":            "Hospital",
        "model":            "RandomForest Q-value + ORS Road Distance",
        "recommended":      chosen["name"],
        "recommended_lat":  chosen["lat"],
        "recommended_lon":  chosen["lon"],
        "recommended_beds": chosen["free_beds"],
        "distance_km":      chosen["distance_km"],
        "drive_min":        eta_to_hospital,
        "dist_source":      chosen["dist_source"],
        "fallback":         fallback["name"],
        "fallback_beds":    fallback["free_beds"],
        "trauma_unit":      chosen["trauma"],
        "trauma_required":  trauma_required,
        "beds_reserved":    beds_reserved,
        "all_hospitals":    hospitals,
        "confidence":       confidence,
        "hospital_candidates": compact_scores(
            scored, ("name", "free_beds", "distance_km", "drive_min", "trauma", "ml_score")
        ),
        "reasoning": (
            f"ORS road routing selected {chosen['name']} "
            f"({chosen['distance_km']}km road, ~{eta_to_hospital} min drive, "
            f"{chosen['free_beds']} beds free, "
            f"trauma={'Yes' if chosen['trauma'] else 'No'}, "
            f"dist_source={chosen['dist_source']}). "
            f"Fallback: {fallback['name']} ({fallback['distance_km']}km, {fallback['free_beds']} beds). "
            f"Reserved {beds_reserved} bed(s) for severity {severity} case."
        ),
    }
