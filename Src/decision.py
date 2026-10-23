import joblib
import pandas as pd
from features import FEATURES
from simulate import SOILS


def recommend(model, soil, w):
    """Combine the ML prediction with safety rules.
    Returns (message, water_mm)."""
    row = {
        "temperature_2m_max": w["tmax"],
        "temperature_2m_min": w["tmin"],
        "precipitation_sum": w["rain"],
        "et0_fao_evapotranspiration": w["et0"],
        "windspeed_10m_max": w["wind"],
        "shortwave_radiation_sum": w["rad"],
        "kc": w["kc"],
        "soil_is_sandy": 1 if soil == "sandy" else 0,
        "rain_last3": w["rain_last3"],
        "et0_last3": w["et0_last3"],
        "rain_next2": w["rain_next2"],
        "prev_depletion": w["prev_depletion"],
    }
    X = pd.DataFrame([row])[FEATURES]
    ml_says_irrigate = int(model.predict(X)[0])

    taw = SOILS[soil]["fc"] - SOILS[soil]["wp"]
    dep = w["prev_depletion"]
    water = round(dep * taw + w["et0"] * w["kc"], 1)

    if dep >= 0.9:
        return "Irrigate now: critically dry", water
    if w["rain_next2"] >= 5:
        return "Skip: rain expected in next 2 days", 0
    if ml_says_irrigate:
        return "Irrigate (model recommendation)", water
    return "No irrigation needed", 0


if __name__ == "__main__":
    model = joblib.load("model.pkl")
    base = dict(tmax=30, tmin=15, rain=0, et0=4, wind=8, rad=18, kc=1.15,
                rain_last3=0, et0_last3=4, rain_next2=0, prev_depletion=0.1)
    tests = {
        "Normal day": dict(base),
        "Rain coming": dict(base, rain_next2=12, prev_depletion=0.4),
        "Hot and dry": dict(base, tmax=38, et0=7, et0_last3=7, prev_depletion=0.6),
        "Critically dry": dict(base, prev_depletion=0.95),
    }
    for soil in ["loamy", "sandy"]:
        for name, w in tests.items():
            msg, water = recommend(model, soil, w)
            print(f"{soil:6} | {name:15} | {msg} | {water} mm")