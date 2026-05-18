"""
data.py — uses real Bengaluru hospital & ambulance data from datasets/
"""
import csv
import json
import math
import os

BASE_DIR = os.path.dirname(__file__)

# ---------------------------------------------------------------------------
# Load real hospital coordinates from geojson
# ---------------------------------------------------------------------------
_HOSPITAL_COORDS = {}
_geojson_path = os.path.join(BASE_DIR, "datasets", "bengaluru_hospitals.geojson")
with open(_geojson_path, "r", encoding="utf-8") as _f:
    _geojson = json.load(_f)
for _feat in _geojson["features"]:
    _name = _feat["properties"].get("name", "")
    _geom = _feat.get("geometry", {})
    if _geom and _geom.get("type") == "Point":
        _lon, _lat = _geom["coordinates"]
        _HOSPITAL_COORDS[_name] = {"lat": _lat, "lon": _lon}

# Known name mappings (CSV name → geojson name)
_NAME_MAP = {
    "M S Ramaiah Hospital": "M S Ramaiah Hospital",
    "Vydehi Hospital": "Vydehi Institute Of Medical Sciences And Research Centre",
    "MS Ramaiah Memorial Hospital": "M S Ramaiah Hospital",
    "Rajarajeswari Medical College And Hospital": "Rajarajeswari Medical College And Hospital",
    "Sapthagiri Super Speciality Hospital": "Sapthagiri Hospital",
    "St. John's Medical Centre": "St.John's Medical College Hospital",
    "St Johns Medical College Hospital": "St.John's Medical College Hospital",
    "Manipal Hospital": "Manipal Hospital",
    "Manipal Hospital Old Airport Road": "Manipal Hospital",
    "Fortis Hospital": "Fortis Hospitals, Bannerghatta Road",
    "Fortis Hospital Bannerghatta Road": "Fortis Hospitals, Bannerghatta Road",
    "Apollo Hospital": "Apollo Hospital",
    "Apollo Hospital Bannerghatta": "Apollo Hospital",
    "Columbia Asia Hebbal": "Columbia Asia Referral Hospital",
    "Sakra World Hospital": "Sakra World Hospital",
    "Narayana Health City": "Narayana Hrudayalaya",
    "Aster CMI Hospital": "Aster CMI Hospital",
}

# Hardcoded real coords for hospitals not matched in GeoJSON
_HARDCODED_COORDS = {
    "Rajarajeswari Medical College And Hospital": {"lat": 12.9139, "lon": 77.4846},
    "PES University Institute Of Medical Sciences And Research": {"lat": 12.9352, "lon": 77.5356},
    "BGS Medical College and Hospital": {"lat": 12.8182, "lon": 77.5134},
    "Cytecare hospitals private ltd": {"lat": 13.0656, "lon": 77.5837},
    "Sanjeevini Hospital": {"lat": 12.9908, "lon": 77.5697},
    "Dr B R Ambedkar Medical College And Hospital": {"lat": 13.0200, "lon": 77.5930},
    "Columbia Asia Hebbal": {"lat": 13.0459, "lon": 77.5965},
    "Narayana Health City": {"lat": 12.8960, "lon": 77.6090},
}

def _get_coords(name):
    geojson_name = _NAME_MAP.get(name, name)
    if geojson_name in _HOSPITAL_COORDS:
        return _HOSPITAL_COORDS[geojson_name]
    if name in _HARDCODED_COORDS:
        return _HARDCODED_COORDS[name]
    return _HOSPITAL_COORDS.get(name, {"lat": 12.9716, "lon": 77.5946})

# ---------------------------------------------------------------------------
# Load real hospitals from CSV
# ---------------------------------------------------------------------------
_CAPACITY_MAP = {
    "M S Ramaiah Hospital": 120,
    "MS Ramaiah Memorial Hospital": 90,
    "Vydehi Hospital": 95,
    "Rajarajeswari Medical College And Hospital": 110,
    "Sapthagiri Super Speciality Hospital": 80,
    "New Varalakshmi Hospital": 60,
    "PES University Institute Of Medical Sciences And Research": 70,
    "BGS Medical College and Hospital": 85,
    "Cytecare hospitals private ltd": 50,
    "Sanjeevini Hospital": 45,
    "Dr B R Ambedkar Medical College And Hospital": 100,
    "Spine Care And Ortho Care Hospital": 40,
    "Fortis Hospital Bannerghatta Road": 280,
    "Apollo Hospital Bannerghatta": 250,
    "Columbia Asia Hebbal": 150,
    "Manipal Hospital Old Airport Road": 300,
    "Sakra World Hospital": 200,
    "St Johns Medical College Hospital": 400,
    "Narayana Health City": 350,
    "Aster CMI Hospital": 180,
}

HOSPITALS = []
_csv_path = os.path.join(BASE_DIR, "datasets", "curated_emergency_hospitals.csv")
with open(_csv_path) as _f:
    for row in csv.DictReader(_f):
        name = row["name"]
        coords = _get_coords(name)
        HOSPITALS.append({
            "name": name,
            "capacity": _CAPACITY_MAP.get(name, 60),
            "trauma": row["emergency"].strip().lower() == "true" or row["polytrauma"].strip().lower() == "true",
            "emergency": row["emergency"].strip().lower() == "true",
            "polytrauma": row["polytrauma"].strip().lower() == "true",
            "neurosurgery": row["neurosurgery"].strip().lower() == "true",
            "orthopaedics": row["orthopaedics"].strip().lower() == "true",
            "accident_victims": row["accident_victims"].strip().lower() == "true",
            "lat": coords["lat"],
            "lon": coords["lon"],
        })

# ---------------------------------------------------------------------------
# Load real ambulance bases from CSV
# ---------------------------------------------------------------------------
AMBULANCE_BASES = []
_amb_path = os.path.join(BASE_DIR, "datasets", "ambulance_bases.csv")
with open(_amb_path) as _f:
    for row in csv.DictReader(_f):
        AMBULANCE_BASES.append({
            "id": row["id"],
            "name": row["name"],
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "zone": row["zone"],
            "capability": 0.88,
            "load": 0.25,
        })

# ---------------------------------------------------------------------------
# Haversine distance (km)
# ---------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

# ---------------------------------------------------------------------------
# Locations with coordinates + profiles
# ---------------------------------------------------------------------------
LOCATIONS = [
    "MG Road",
    "Brigade Road",
    "Residency Road",
    "Koramangala",
    "Indiranagar",
    "Whitefield",
    "Hebbal",
    "Silk Board",
    "Marathahalli",
    "Electronic City",
    "Yeshwantpur",
    "Bannerghatta Road",
    "Rajajinagar",
    "Jayanagar",
    "JP Nagar",
    "Mysore Road",
    "Tumkur Road",
    "Sarjapur Road",
    "Old Airport Road",
    "KR Puram",
    "Nagavara",
]

LOCATION_COORDS = {
    "MG Road":           {"lat": 12.9757, "lon": 77.6011},
    "Brigade Road":      {"lat": 12.9740, "lon": 77.6076},
    "Residency Road":    {"lat": 12.9716, "lon": 77.5946},
    "Koramangala":       {"lat": 12.9352, "lon": 77.6245},
    "Indiranagar":       {"lat": 12.9784, "lon": 77.6408},
    "Whitefield":        {"lat": 12.9698, "lon": 77.7499},
    "Hebbal":            {"lat": 13.0358, "lon": 77.5970},
    "Silk Board":        {"lat": 12.9177, "lon": 77.6227},
    "Marathahalli":      {"lat": 12.9591, "lon": 77.7004},
    "Electronic City":   {"lat": 12.8399, "lon": 77.6770},
    "Yeshwantpur":       {"lat": 13.0213, "lon": 77.5541},
    "Bannerghatta Road": {"lat": 12.8946, "lon": 77.5966},
    "Rajajinagar":       {"lat": 12.9902, "lon": 77.5558},
    "Jayanagar":         {"lat": 12.9308, "lon": 77.5838},
    "JP Nagar":          {"lat": 12.9063, "lon": 77.5857},
    "Mysore Road":       {"lat": 12.9519, "lon": 77.5204},
    "Tumkur Road":       {"lat": 13.0150, "lon": 77.5200},
    "Sarjapur Road":     {"lat": 12.9121, "lon": 77.6800},
    "Old Airport Road":  {"lat": 12.9591, "lon": 77.6480},
    "KR Puram":          {"lat": 13.0050, "lon": 77.6950},
    "Nagavara":          {"lat": 13.0450, "lon": 77.6210},
}

LOCATION_PROFILES = {
    "MG Road":           {"traffic_risk": 0.72, "access_complexity": 0.55, "zone": "central"},
    "Brigade Road":      {"traffic_risk": 0.78, "access_complexity": 0.62, "zone": "central"},
    "Residency Road":    {"traffic_risk": 0.58, "access_complexity": 0.42, "zone": "central"},
    "Koramangala":       {"traffic_risk": 0.68, "access_complexity": 0.50, "zone": "south"},
    "Indiranagar":       {"traffic_risk": 0.60, "access_complexity": 0.40, "zone": "east"},
    "Whitefield":        {"traffic_risk": 0.82, "access_complexity": 0.70, "zone": "east"},
    "Hebbal":            {"traffic_risk": 0.74, "access_complexity": 0.58, "zone": "north"},
    "Silk Board":        {"traffic_risk": 0.92, "access_complexity": 0.82, "zone": "south"},
    "Marathahalli":      {"traffic_risk": 0.86, "access_complexity": 0.74, "zone": "east"},
    "Electronic City":   {"traffic_risk": 0.88, "access_complexity": 0.72, "zone": "south"},
    "Yeshwantpur":       {"traffic_risk": 0.76, "access_complexity": 0.60, "zone": "north"},
    "Bannerghatta Road": {"traffic_risk": 0.80, "access_complexity": 0.65, "zone": "south"},
    "Rajajinagar":       {"traffic_risk": 0.74, "access_complexity": 0.58, "zone": "west"},
    "Jayanagar":         {"traffic_risk": 0.68, "access_complexity": 0.52, "zone": "south"},
    "JP Nagar":          {"traffic_risk": 0.70, "access_complexity": 0.55, "zone": "south"},
    "Mysore Road":       {"traffic_risk": 0.82, "access_complexity": 0.70, "zone": "west"},
    "Tumkur Road":       {"traffic_risk": 0.78, "access_complexity": 0.64, "zone": "north"},
    "Sarjapur Road":     {"traffic_risk": 0.85, "access_complexity": 0.75, "zone": "east"},
    "Old Airport Road":  {"traffic_risk": 0.72, "access_complexity": 0.58, "zone": "east"},
    "KR Puram":          {"traffic_risk": 0.80, "access_complexity": 0.66, "zone": "east"},
    "Nagavara":          {"traffic_risk": 0.70, "access_complexity": 0.54, "zone": "north"},
}

# ---------------------------------------------------------------------------
# Dynamic distance calculators (replace old lookup tables)
# ---------------------------------------------------------------------------
def get_hospital_distance(location: str, hospital_name: str) -> float:
    loc_coords = LOCATION_COORDS.get(location, LOCATION_COORDS["MG Road"])
    hosp = next((h for h in HOSPITALS if h["name"] == hospital_name), None)
    if hosp:
        return round(haversine(loc_coords["lat"], loc_coords["lon"], hosp["lat"], hosp["lon"]), 1)
    return 9.0

def get_ambulance_distance(base_id: str, location: str) -> float:
    base = next((b for b in AMBULANCE_BASES if b["id"] == base_id), None)
    loc_coords = LOCATION_COORDS.get(location, LOCATION_COORDS["MG Road"])
    if base:
        return round(haversine(base["lat"], base["lon"], loc_coords["lat"], loc_coords["lon"]), 1)
    return 8.0

# ---------------------------------------------------------------------------
# Keep these for backward compatibility with existing agent code
# ---------------------------------------------------------------------------
HOSPITAL_DISTANCE_KM = {
    loc: {h["name"]: get_hospital_distance(loc, h["name"]) for h in HOSPITALS}
    for loc in LOCATIONS
}

AMBULANCE_BASE_DISTANCE_KM = {
    base["id"]: {loc: get_ambulance_distance(base["id"], loc) for loc in LOCATIONS}
    for base in AMBULANCE_BASES
}

ROUTES_BY_LOCATION = {
    "MG Road": [
        {"name": "Brigade Road", "traffic_fit": 0.70, "signal_count": 3, "critical_access": 0.86},
        {"name": "Residency Road", "traffic_fit": 0.84, "signal_count": 2, "critical_access": 0.68},
        {"name": "Cubbon corridor", "traffic_fit": 0.76, "signal_count": 3, "critical_access": 0.72},
    ],
    "Silk Board": [
        {"name": "Hosur Road corridor", "traffic_fit": 0.90, "signal_count": 5, "critical_access": 0.92},
        {"name": "BTM service road", "traffic_fit": 0.74, "signal_count": 3, "critical_access": 0.70},
        {"name": "Outer Ring Road feeder", "traffic_fit": 0.82, "signal_count": 4, "critical_access": 0.80},
    ],
    "Whitefield": [
        {"name": "ITPL Main Road", "traffic_fit": 0.78, "signal_count": 4, "critical_access": 0.76},
        {"name": "Varthur Road", "traffic_fit": 0.72, "signal_count": 5, "critical_access": 0.82},
        {"name": "Old Airport Road", "traffic_fit": 0.85, "signal_count": 4, "critical_access": 0.74},
    ],
    "Indiranagar": [
        {"name": "100 Feet Road", "traffic_fit": 0.82, "signal_count": 3, "critical_access": 0.72},
        {"name": "Old Airport Road", "traffic_fit": 0.88, "signal_count": 4, "critical_access": 0.84},
        {"name": "CMH Road", "traffic_fit": 0.68, "signal_count": 2, "critical_access": 0.58},
    ],
    "Koramangala": [
        {"name": "Sony World Junction", "traffic_fit": 0.76, "signal_count": 4, "critical_access": 0.74},
        {"name": "Sarjapur Road", "traffic_fit": 0.82, "signal_count": 4, "critical_access": 0.80},
        {"name": "Adugodi corridor", "traffic_fit": 0.70, "signal_count": 3, "critical_access": 0.68},
    ],
    "Hebbal": [
        {"name": "Airport Road", "traffic_fit": 0.84, "signal_count": 4, "critical_access": 0.86},
        {"name": "Outer Ring Road north", "traffic_fit": 0.78, "signal_count": 3, "critical_access": 0.74},
        {"name": "Bellary Road service lane", "traffic_fit": 0.66, "signal_count": 2, "critical_access": 0.62},
    ],
    "Marathahalli": [
        {"name": "Outer Ring Road east", "traffic_fit": 0.86, "signal_count": 5, "critical_access": 0.82},
        {"name": "HAL corridor", "traffic_fit": 0.74, "signal_count": 4, "critical_access": 0.72},
        {"name": "Varthur slip road", "traffic_fit": 0.70, "signal_count": 3, "critical_access": 0.66},
    ],
    "Brigade Road": [
        {"name": "MG Road", "traffic_fit": 0.72, "signal_count": 3, "critical_access": 0.80},
        {"name": "Residency Road", "traffic_fit": 0.80, "signal_count": 2, "critical_access": 0.70},
        {"name": "Cubbon corridor", "traffic_fit": 0.75, "signal_count": 3, "critical_access": 0.72},
    ],
    "Residency Road": [
        {"name": "MG Road", "traffic_fit": 0.76, "signal_count": 2, "critical_access": 0.78},
        {"name": "Brigade Road", "traffic_fit": 0.72, "signal_count": 3, "critical_access": 0.74},
        {"name": "Lavelle Road", "traffic_fit": 0.82, "signal_count": 2, "critical_access": 0.68},
    ],
    "Electronic City": [
        {"name": "Hosur Road flyover", "traffic_fit": 0.88, "signal_count": 5, "critical_access": 0.90},
        {"name": "NICE Road connector", "traffic_fit": 0.80, "signal_count": 3, "critical_access": 0.78},
        {"name": "Electronic City Phase 2 road", "traffic_fit": 0.72, "signal_count": 4, "critical_access": 0.70},
    ],
    "Yeshwantpur": [
        {"name": "Tumkur Road", "traffic_fit": 0.78, "signal_count": 4, "critical_access": 0.80},
        {"name": "Chord Road", "traffic_fit": 0.74, "signal_count": 3, "critical_access": 0.72},
        {"name": "Outer Ring Road west", "traffic_fit": 0.82, "signal_count": 4, "critical_access": 0.76},
    ],
    "Bannerghatta Road": [
        {"name": "JP Nagar service road", "traffic_fit": 0.76, "signal_count": 4, "critical_access": 0.78},
        {"name": "NICE Road south", "traffic_fit": 0.84, "signal_count": 3, "critical_access": 0.82},
        {"name": "Arekere junction bypass", "traffic_fit": 0.70, "signal_count": 3, "critical_access": 0.68},
    ],
    "Rajajinagar": [
        {"name": "Chord Road", "traffic_fit": 0.80, "signal_count": 3, "critical_access": 0.78},
        {"name": "Mysore Road feeder", "traffic_fit": 0.76, "signal_count": 4, "critical_access": 0.74},
        {"name": "Rajajinagar Industrial area road", "traffic_fit": 0.68, "signal_count": 2, "critical_access": 0.62},
    ],
    "Jayanagar": [
        {"name": "11th Main Road", "traffic_fit": 0.78, "signal_count": 3, "critical_access": 0.74},
        {"name": "Bannerghatta Road connector", "traffic_fit": 0.74, "signal_count": 4, "critical_access": 0.78},
        {"name": "KR Road", "traffic_fit": 0.70, "signal_count": 3, "critical_access": 0.66},
    ],
    "JP Nagar": [
        {"name": "Bannerghatta Road", "traffic_fit": 0.80, "signal_count": 4, "critical_access": 0.82},
        {"name": "NICE Road JP Nagar", "traffic_fit": 0.86, "signal_count": 3, "critical_access": 0.84},
        {"name": "Kanakapura Road slip", "traffic_fit": 0.72, "signal_count": 3, "critical_access": 0.70},
    ],
    "Mysore Road": [
        {"name": "NICE Road west", "traffic_fit": 0.84, "signal_count": 3, "critical_access": 0.86},
        {"name": "Kengeri bypass", "traffic_fit": 0.76, "signal_count": 4, "critical_access": 0.74},
        {"name": "Outer Ring Road southwest", "traffic_fit": 0.80, "signal_count": 4, "critical_access": 0.78},
    ],
    "Tumkur Road": [
        {"name": "Yeshwantpur connector", "traffic_fit": 0.78, "signal_count": 4, "critical_access": 0.76},
        {"name": "Peenya Industrial bypass", "traffic_fit": 0.72, "signal_count": 3, "critical_access": 0.70},
        {"name": "Outer Ring Road northwest", "traffic_fit": 0.82, "signal_count": 4, "critical_access": 0.80},
    ],
    "Sarjapur Road": [
        {"name": "Outer Ring Road southeast", "traffic_fit": 0.84, "signal_count": 5, "critical_access": 0.82},
        {"name": "Carmelaram Road", "traffic_fit": 0.74, "signal_count": 3, "critical_access": 0.72},
        {"name": "Harlur Road connector", "traffic_fit": 0.70, "signal_count": 3, "critical_access": 0.68},
    ],
    "Old Airport Road": [
        {"name": "HAL Old Airport Road", "traffic_fit": 0.82, "signal_count": 4, "critical_access": 0.84},
        {"name": "Domlur flyover", "traffic_fit": 0.78, "signal_count": 3, "critical_access": 0.76},
        {"name": "Indiranagar connector", "traffic_fit": 0.74, "signal_count": 3, "critical_access": 0.72},
    ],
    "KR Puram": [
        {"name": "Old Madras Road", "traffic_fit": 0.80, "signal_count": 4, "critical_access": 0.82},
        {"name": "Outer Ring Road KR Puram", "traffic_fit": 0.84, "signal_count": 5, "critical_access": 0.80},
        {"name": "Tin Factory junction road", "traffic_fit": 0.70, "signal_count": 3, "critical_access": 0.68},
    ],
    "Nagavara": [
        {"name": "HBR Layout connector", "traffic_fit": 0.76, "signal_count": 3, "critical_access": 0.74},
        {"name": "Outer Ring Road north", "traffic_fit": 0.82, "signal_count": 4, "critical_access": 0.80},
        {"name": "Thanisandra Road", "traffic_fit": 0.72, "signal_count": 3, "critical_access": 0.70},
    ],
}

