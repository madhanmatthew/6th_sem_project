import random

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


TRAFFIC_FEATURES = [
    "severity",
    "injured",
    "traffic_heavy",
    "weather_rain",
    "vehicle_risk",
    "location_risk",
    "incident_risk",
    "hour",
]

HOSPITAL_FEATURES = [
    "severity",
    "injured",
    "free_beds",
    "trauma",
    "distance_km",
    "load_ratio",
]

VEHICLE_RISK = {
    "unknown": 0.25,
    "bike": 0.35,
    "motorcycle": 0.38,
    "auto": 0.42,
    "car": 0.48,
    "van": 0.58,
    "bus": 0.78,
    "truck": 0.88,
}

INCIDENT_RISK = {
    "road accident": 0.45,
    "traffic collision": 0.72,
    "fire emergency": 0.82,
    "medical emergency": 0.38,
}

SEVERITY_LABELS = {
    1: "Low",
    2: "Minor",
    3: "Moderate",
    4: "High",
    5: "Critical",
}


def train_severity_classifier():
    random.seed(42)
    templates = {
        1: [
            "light scratch on {loc}, {injured} injured, {vehicle} involved",
            "tiny collision near {loc}, {injured} injured, {vehicle} moving slowly",
            "low impact incident at {loc}, {injured} injured, {vehicle} involved",
        ],
        2: [
            "minor accident near {loc}, {injured} injured, {vehicle} collision",
            "small crash on {loc}, {injured} injured, {vehicle} hit divider",
            "fender bender at {loc}, {injured} injured, {vehicle} involved",
        ],
        3: [
            "accident at {loc}, {injured} injured, {vehicle} collision",
            "moderate crash near {loc}, {injured} injured, {vehicle} damaged",
            "road accident on {loc}, {injured} injured, {vehicle} involved",
        ],
        4: [
            "major crash on {loc}, {injured} injured, {vehicle} collision",
            "truck accident at {loc}, {injured} injured, road blocked",
            "bus collision near {loc}, {injured} injured, heavy traffic",
        ],
        5: [
            "critical rollover at {loc}, {injured} injured, {vehicle} blocking road",
            "fatal severe accident near {loc}, multiple injured, {vehicle} collision",
            "serious head-on crash at {loc}, {injured} injured, emergency help needed",
        ],
    }
    injured_ranges = {
        1: (1, 1),
        2: (1, 2),
        3: (1, 3),
        4: (3, 5),
        5: (5, 8),
    }
    locations = ["MG Road", "Silk Board", "Whitefield", "Indiranagar", "Hebbal"]
    vehicles = ["car", "bike", "truck", "bus", "van", "auto"]
    rows = []

    for severity in range(1, 6):
        for _ in range(180):
            low, high = injured_ranges[severity]
            injured = random.randint(low, high)
            vehicle_pool = vehicles if severity < 4 else ["truck", "bus", "car", "van"]
            vehicle = random.choice(vehicle_pool)
            text = random.choice(templates[severity]).format(
                loc=random.choice(locations),
                injured=injured,
                vehicle=vehicle,
            )
            if severity >= 4 and random.random() < 0.35:
                text += " road blocked"
            if random.random() < 0.22:
                text += " heavy traffic"
            if random.random() < 0.18:
                text += " rain"
            rows.append({"text": text, "severity": severity})

    for _ in range(120):
        severity = random.choice([2, 3, 4])
        low, high = injured_ranges[severity]
        injured = random.randint(low, high)
        vehicle = random.choice(vehicles)
        text = random.choice(templates[severity]).format(
            loc=random.choice(locations),
            injured=injured,
            vehicle=vehicle,
        )
        text = text.replace("accident", random.choice(["incident", "collision", "crash"]))
        rows.append({"text": text, "severity": severity})

    df = pd.DataFrame(rows)
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"], df["severity"], test_size=0.22, random_state=42, stratify=df["severity"]
    )
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    accuracy = accuracy_score(y_test, preds)
    return model, {"accuracy": round(float(accuracy), 3), "samples": len(df)}


def train_traffic_models():
    np.random.seed(43)
    n = 900
    df = pd.DataFrame(
        {
            "severity": np.random.randint(1, 6, n),
            "injured": np.random.randint(1, 9, n),
            "traffic_heavy": np.random.randint(0, 2, n),
            "weather_rain": np.random.randint(0, 2, n),
            "vehicle_risk": np.random.uniform(0.25, 0.9, n),
            "location_risk": np.random.uniform(0.35, 0.95, n),
            "incident_risk": np.random.uniform(0.35, 0.9, n),
            "hour": np.random.randint(0, 24, n),
        }
    )
    rush_hour = df["hour"].between(8, 11) | df["hour"].between(17, 20)
    df["delay_min"] = (
        df["severity"] * 2.2
        + df["injured"] * 0.5
        + df["traffic_heavy"] * 8.0
        + df["weather_rain"] * 3.2
        + df["vehicle_risk"] * 3.0
        + df["location_risk"] * 3.0
        + df["incident_risk"] * 2.0
        + rush_hour.astype(int) * 2.5
        + np.random.normal(0, 1.8, n)
    ).clip(0, 35)
    df["congestion_class"] = pd.cut(
        df["delay_min"], bins=[-1, 6, 13, 21, 40], labels=[0, 1, 2, 3]
    ).astype(int)

    x = df[TRAFFIC_FEATURES]
    y_delay = df["delay_min"]
    y_class = df["congestion_class"]
    x_train, x_test, y_delay_train, y_delay_test, y_class_train, y_class_test = train_test_split(
        x, y_delay, y_class, test_size=0.22, random_state=42, stratify=y_class
    )

    delay_model = RandomForestRegressor(n_estimators=120, random_state=42, max_depth=8)
    class_model = RandomForestClassifier(n_estimators=120, random_state=42, max_depth=8)
    delay_model.fit(x_train, y_delay_train)
    class_model.fit(x_train, y_class_train)

    delay_preds = delay_model.predict(x_test)
    class_preds = class_model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_delay_test, delay_preds))
    accuracy = accuracy_score(y_class_test, class_preds)
    return (
        {"delay": delay_model, "class": class_model},
        {"rmse_minutes": round(float(rmse), 2), "accuracy": round(float(accuracy), 3), "samples": n},
    )


def train_hospital_value_model():
    np.random.seed(44)
    n = 900
    df = pd.DataFrame(
        {
            "severity": np.random.randint(1, 6, n),
            "injured": np.random.randint(1, 9, n),
            "free_beds": np.random.randint(0, 75, n),
            "trauma": np.random.randint(0, 2, n),
            "distance_km": np.random.uniform(2.0, 16.0, n),
            "load_ratio": np.random.uniform(0.25, 0.95, n),
        }
    )
    trauma_needed = (df["severity"] >= 3).astype(int)
    capacity_fit = np.minimum(1.0, df["free_beds"] / df["injured"].clip(lower=1))
    df["q_value"] = (
        capacity_fit * 0.2
        + (df["free_beds"] / 75) * 0.16
        + np.where((df["trauma"] == 1) | (trauma_needed == 0), 0.22, 0.05)
        + (1 - ((df["distance_km"] - 2) / 14).clip(0, 1)) * 0.3
        + (1 - df["load_ratio"]) * 0.1
        + (df["severity"] / 5) * 0.06
        + np.random.normal(0, 0.025, n)
    ).clip(0, 1)

    x = df[HOSPITAL_FEATURES]
    y = df["q_value"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.22, random_state=42
    )
    model = RandomForestRegressor(n_estimators=140, random_state=42, max_depth=9)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return model, {"rmse_q_value": round(float(rmse), 3), "samples": n}


def train_ml_models():
    severity_model, severity_metrics = train_severity_classifier()
    traffic_models, traffic_metrics = train_traffic_models()
    hospital_model, hospital_metrics = train_hospital_value_model()
    print(f"[ML] Severity classifier accuracy = {severity_metrics['accuracy']:.3f}")
    print(f"[ML] Traffic model RMSE = {traffic_metrics['rmse_minutes']:.2f} min")
    print(f"[ML] Hospital value model RMSE = {hospital_metrics['rmse_q_value']:.3f}")
    return (
        {
            "severity": severity_model,
            "traffic": traffic_models,
            "hospital": hospital_model,
        },
        {
            "severity": severity_metrics,
            "traffic": traffic_metrics,
            "hospital": hospital_metrics,
        },
    )


def apply_severity_model(parsed: dict, severity_model) -> dict:
    probabilities = severity_model.predict_proba([parsed["raw"]])[0]
    classes = severity_model.named_steps["clf"].classes_
    ml_severity = int(classes[int(np.argmax(probabilities))])
    confidence = round(float(np.max(probabilities)), 2)

    ML_CONFIDENCE_THRESHOLD = 0.55
    if confidence >= ML_CONFIDENCE_THRESHOLD:
        # ML is confident — use ML prediction directly (not max, actual override)
        final_severity = ml_severity
        severity_source = "ml_model"
    else:
        # ML not confident enough — keep rule-based value as authoritative fallback
        final_severity = parsed["severity"]
        severity_source = "rule_based"

    final_severity = max(1, min(5, final_severity))
    parsed.update(
        {
            "rule_severity": parsed["severity"],
            "ml_severity": ml_severity,
            "ml_severity_confidence": confidence,
            "severity": final_severity,
            "severity_label": SEVERITY_LABELS[final_severity],
            "severity_source": severity_source,
        }
    )
    return parsed


def traffic_feature_frame(parsed: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "severity": parsed["severity"],
                "injured": parsed["injured"],
                "traffic_heavy": int(parsed.get("traffic_hint") == "heavy"),
                "weather_rain": int(parsed.get("weather_hint") == "rain"),
                "vehicle_risk": VEHICLE_RISK.get(parsed.get("vehicle", "unknown"), 0.25),
                "location_risk": parsed.get("location_risk", 0.55),
                "incident_risk": INCIDENT_RISK.get(parsed.get("incident_type", "road accident"), 0.45),
                "hour": 9,
            }
        ],
        columns=TRAFFIC_FEATURES,
    )


def hospital_feature_frame(
    severity: int,
    injured: int,
    free_beds: int,
    trauma: bool,
    distance_km: float,
    load_ratio: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "severity": severity,
                "injured": injured,
                "free_beds": free_beds,
                "trauma": int(trauma),
                "distance_km": distance_km,
                "load_ratio": load_ratio,
            }
        ],
        columns=HOSPITAL_FEATURES,
    )
