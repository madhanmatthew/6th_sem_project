"""
routing.py — Real road distance via OpenRouteService Matrix API
Falls back to haversine × 1.4 if API key not set or call fails.

Usage:
    Set ORS_API_KEY environment variable, OR paste key into ORS_KEY below.
    Replace "PASTE_YOUR_ORS_KEY_HERE" with your actual key from:
    https://openrouteservice.org/dev/#/api-docs
"""

import datetime
import os
from math import atan2, cos, radians, sin, sqrt

import requests

# ── API KEY ─────────────────────────────────────────────────────────────────
# Priority: env variable → hardcoded below
ORS_KEY = os.getenv("ORS_API_KEY", "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImM1YTdjZmQzZjljNjQzZmJhMzQ4YjAyNjQ2YTkwMzU2IiwiaCI6Im11cm11cjY0In0=")

ORS_MATRIX_URL     = "https://api.openrouteservice.org/v2/matrix/driving-car"
ORS_DIRECTIONS_URL = "https://api.openrouteservice.org/v2/directions/driving-car"
REQUEST_TIMEOUT    = 6  # seconds

# ── SPEED TABLE (Bengaluru realistic — base free-flow, before congestion) ────
# These are raw drive speeds used only for haversine fallback duration estimate.
# Ambulance agent applies additional Bengaluru congestion multiplier on top.
_SPEED = {"peak": 18, "normal": 28, "night": 38}

def _speed_kmph(hour: int = None) -> int:
    h = hour if hour is not None else datetime.datetime.now().hour
    if h in range(8, 11) or h in range(17, 21):
        return _SPEED["peak"]
    if h in range(0, 6) or h == 23:
        return _SPEED["night"]
    return _SPEED["normal"]

# ── HAVERSINE FALLBACK ───────────────────────────────────────────────────────
def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    la1, lo1, la2, lo2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((la2 - la1) / 2) ** 2 + cos(la1) * cos(la2) * sin((lo2 - lo1) / 2) ** 2
    return round(R * 2 * atan2(sqrt(a), sqrt(1 - a)), 2)

def _fallback(lat1, lon1, lat2, lon2) -> dict:
    """
    Haversine × 1.45 road-correction factor (Bengaluru roads are winding).
    Duration based on realistic base speeds; ambulance agent adds congestion
    multiplier on top of this. Minimum road distance 1.0km (mobilization gap).
    """
    dist = round(_haversine_km(lat1, lon1, lat2, lon2) * 1.45, 2)
    dist = max(1.0, dist)          # even same-zone base has ~1km internal distance
    speed = _speed_kmph()
    dur = round((dist / speed) * 60, 1)
    dur = max(2.0, dur)
    return {"distance_km": dist, "duration_min": dur, "source": "haversine_fallback"}

# ── IN-MEMORY CACHE ──────────────────────────────────────────────────────────
_cache: dict = {}

def _cache_key(*coords) -> tuple:
    return tuple(round(c, 4) for c in coords)

# ── SINGLE PAIR ──────────────────────────────────────────────────────────────
def get_road_distance(orig_lat, orig_lon, dest_lat, dest_lon) -> dict:
    """
    Returns {distance_km, duration_min, source}
    source = "ors" | "haversine_fallback"
    """
    key = _cache_key(orig_lat, orig_lon, dest_lat, dest_lon)
    if key in _cache:
        return _cache[key]

    if ORS_KEY == "PASTE_YOUR_ORS_KEY_HERE":
        result = _fallback(orig_lat, orig_lon, dest_lat, dest_lon)
        _cache[key] = result
        return result

    try:
        payload = {
            "locations": [[orig_lon, orig_lat], [dest_lon, dest_lat]],
            "metrics": ["distance", "duration"],
            "units": "km",
        }
        headers = {"Authorization": ORS_KEY, "Content-Type": "application/json"}
        resp = requests.post(ORS_MATRIX_URL, json=payload,
                             headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        result = {
            "distance_km": round(data["distances"][0][1], 2),
            "duration_min": round(data["durations"][0][1] / 60, 1),
            "source": "ors",
        }
    except Exception:
        result = _fallback(orig_lat, orig_lon, dest_lat, dest_lon)

    _cache[key] = result
    return result

# ── BATCH: ONE CALL FOR ALL HOSPITALS ────────────────────────────────────────
def batch_hospital_distances(accident_lat, accident_lon, hospitals: list) -> dict:
    """
    Single ORS matrix call: accident location → all hospitals.
    Returns {hospital_name: {distance_km, duration_min, source}}

    This uses 1 API call instead of 1 per hospital — efficient.
    Falls back per-hospital if API fails.
    """
    if ORS_KEY == "PASTE_YOUR_ORS_KEY_HERE":
        return {
            h["name"]: _fallback(accident_lat, accident_lon, h["lat"], h["lon"])
            for h in hospitals
        }

    # Build locations list: [accident] + [all hospitals]
    locations = [[accident_lon, accident_lat]]
    for h in hospitals:
        locations.append([h["lon"], h["lat"]])

    try:
        payload = {
            "locations": locations,
            "metrics": ["distance", "duration"],
            "units": "km",
            "sources": [0],
            "destinations": list(range(1, len(hospitals) + 1)),
        }
        headers = {"Authorization": ORS_KEY, "Content-Type": "application/json"}
        resp = requests.post(ORS_MATRIX_URL, json=payload,
                             headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        result = {}
        for i, h in enumerate(hospitals):
            d = data["distances"][0][i]
            t = data["durations"][0][i]
            # ORS returns null for unreachable locations
            if d is None or t is None:
                result[h["name"]] = _fallback(
                    accident_lat, accident_lon, h["lat"], h["lon"]
                )
            else:
                result[h["name"]] = {
                    "distance_km": round(d, 2),
                    "duration_min": round(t / 60, 1),
                    "source": "ors",
                }
        return result

    except Exception:
        # Full fallback: compute haversine for each individually
        return {
            h["name"]: _fallback(accident_lat, accident_lon, h["lat"], h["lon"])
            for h in hospitals
        }

# ── BATCH: ONE CALL FOR ALL AMBULANCE BASES ──────────────────────────────────
def batch_ambulance_distances(accident_lat, accident_lon, bases: list) -> dict:
    """
    Single ORS matrix call: accident location → all ambulance bases.
    Returns {base_id: {distance_km, duration_min, source}}
    """
    if ORS_KEY == "PASTE_YOUR_ORS_KEY_HERE":
        return {
            b["id"]: _fallback(b["lat"], b["lon"], accident_lat, accident_lon)
            for b in bases
        }

    locations = [[accident_lon, accident_lat]]
    for b in bases:
        locations.append([b["lon"], b["lat"]])

    try:
        payload = {
            "locations": locations,
            "metrics": ["distance", "duration"],
            "units": "km",
            "sources": [0],
            "destinations": list(range(1, len(bases) + 1)),
        }
        headers = {"Authorization": ORS_KEY, "Content-Type": "application/json"}
        resp = requests.post(ORS_MATRIX_URL, json=payload,
                             headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        result = {}
        for i, b in enumerate(bases):
            d = data["distances"][0][i]
            t = data["durations"][0][i]
            if d is None or t is None:
                result[b["id"]] = _fallback(
                    b["lat"], b["lon"], accident_lat, accident_lon
                )
            else:
                result[b["id"]] = {
                    "distance_km": round(d, 2),
                    "duration_min": round(t / 60, 1),
                    "source": "ors",
                }
        return result

    except Exception:
        return {
            b["id"]: _fallback(b["lat"], b["lon"], accident_lat, accident_lon)
            for b in bases
        }
