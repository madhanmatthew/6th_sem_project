import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


MODEL_PATH = "eta_model.joblib"
TRAINING_SAMPLES = 500
MODEL_FEATURES = [
    "distance_km",
    "time_of_day",
    "day_of_week",
    "traffic_level",
    "weather",
]


def train_eta_model():
    np.random.seed(42)

    n = TRAINING_SAMPLES
    n_peak     = int(n * 0.40)   # 40% peak hours
    n_nearpeak = int(n * 0.30)   # 30% near-peak
    n_offpeak  = n - n_peak - n_nearpeak  # 30% off-peak

    peak_hours     = [8, 9, 10, 17, 18, 19]
    nearpeak_hours = [7, 11, 16, 20]
    offpeak_hours  = list(range(0, 7)) + list(range(21, 24))

    time_of_day    = (
        np.random.choice(peak_hours,     n_peak)    .tolist() +
        np.random.choice(nearpeak_hours, n_nearpeak).tolist() +
        np.random.choice(offpeak_hours,  n_offpeak) .tolist()
    )
    # Correlated traffic_level: peak → 2-3, near-peak → 1-2, off-peak → 0-1
    traffic_level = (
        np.random.randint(2, 4, n_peak)    .tolist() +
        np.random.randint(1, 3, n_nearpeak).tolist() +
        np.random.randint(0, 2, n_offpeak) .tolist()
    )

    # Shuffle so peak/off-peak aren't all together
    idx = np.random.permutation(n)
    time_of_day   = np.array(time_of_day)[idx]
    traffic_level = np.array(traffic_level)[idx]

    df = pd.DataFrame(
        {
            "distance_km":   np.random.uniform(0.5, 15, n),
            "time_of_day":   time_of_day,
            "day_of_week":   np.random.randint(0, 7, n),
            "traffic_level": traffic_level,
            "weather":       np.random.randint(0, 3, n),
        }
    )
    df["eta_minutes"] = (
        (df["distance_km"] / 45.0) * 60        # base: 45 km/h emergency speed
        + df["traffic_level"] * 2.5             # traffic adds 0-7.5 min
        + df["weather"] * 1.2                   # rain adds 0-2.4 min
        + np.random.normal(0, 1.2, n)           # noise
    ).clip(1.0, 40.0)

    x = df.drop("eta_minutes", axis=1)
    y = df["eta_minutes"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, random_state=42
    )
    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    print(f"[ETA Model] Trained. RMSE = {rmse:.2f} min on held-out test data.")
    importances = dict(zip(MODEL_FEATURES, model.feature_importances_))
    print("[ETA Model] Feature importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        bar = "#" * int(imp * 40)
        print(f"  {feat:<16} {imp:.3f}  {bar}")

    joblib.dump(model, MODEL_PATH)
    return model, rmse
