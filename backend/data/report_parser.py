import re

from data import LOCATION_PROFILES, LOCATIONS


SEVERITY_LABELS = {
    1: "Low",
    2: "Minor",
    3: "Moderate",
    4: "High",
    5: "Critical",
}


def parse_report(text: str) -> dict:
    lower = text.lower()

    severity = 3
    severity_reason = "default moderate incident"
    if any(word in lower for word in ["critical", "fatal", "multiple", "severe", "serious"]):
        severity = 5
        severity_reason = "critical injury keyword detected"
    elif any(word in lower for word in ["major", "truck", "bus", "rollover", "head-on"]):
        severity = 4
        severity_reason = "major collision keyword detected"
    elif any(word in lower for word in ["minor", "fender", "small"]):
        severity = 2
        severity_reason = "minor collision keyword detected"
    elif any(word in lower for word in ["scratch", "tiny", "light"]):
        severity = 1
        severity_reason = "low severity keyword detected"

    nums = re.findall(r"\d+", text)
    injured = int(nums[0]) if nums else 1
    if injured >= 5:
        severity = max(severity, 5)
        severity_reason = "multiple injured count escalated severity"
    elif injured >= 3:
        severity = max(severity, 4)
        severity_reason = "injury count escalated severity"

    location = "MG Road"
    for loc in LOCATIONS:
        if loc.lower() in lower:
            location = loc
            break

    vehicle = "unknown"
    for vehicle_type in ["truck", "bus", "car", "bike", "motorcycle", "auto", "van"]:
        if vehicle_type in lower:
            vehicle = vehicle_type
            break

    incident_type = "road accident"
    if any(word in lower for word in ["fire", "smoke", "burn"]):
        incident_type = "fire emergency"
    elif any(word in lower for word in ["medical", "heart", "stroke", "collapse"]):
        incident_type = "medical emergency"
    elif any(word in lower for word in ["rollover", "collision", "crash", "accident", "hit"]):
        incident_type = "traffic collision"

    traffic_hint = "heavy" if any(
        word in lower for word in ["jam", "blocked", "heavy traffic", "peak", "rush"]
    ) else "normal"
    weather_hint = "rain" if any(
        word in lower for word in ["rain", "wet", "storm", "flood"]
    ) else "clear"
    confidence = min(0.98, 0.62 + (0.08 if location != "MG Road" else 0) + (0.08 if vehicle != "unknown" else 0) + (0.08 if nums else 0))
    location_profile = LOCATION_PROFILES.get(location, LOCATION_PROFILES["MG Road"])

    return {
        "location": location,
        "location_zone": location_profile["zone"],
        "location_risk": location_profile["traffic_risk"],
        "access_complexity": location_profile["access_complexity"],
        "severity": severity,
        "severity_label": SEVERITY_LABELS[severity],
        "severity_reason": severity_reason,
        "injured": injured,
        "vehicle": vehicle,
        "incident_type": incident_type,
        "traffic_hint": traffic_hint,
        "weather_hint": weather_hint,
        "confidence": round(confidence, 2),
        "raw": text,
    }
