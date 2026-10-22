import pandas as pd

# मिट्टी के parameters (mm of water, root zone में)
SOILS = {
    "loamy": {"fc": 180, "wp": 80},   # field capacity, wilting point
    "sandy": {"fc": 120, "wp": 50},
}

def kc_for_day(date):
    """गेहूँ का crop coefficient, बुवाई (1 नवंबर) से दिन गिनकर।"""
    sowing = pd.Timestamp(year=date.year if date.month >= 11 else date.year - 1,
                          month=11, day=1)
    day = (date - sowing).days
    if day < 0 or day > 140:
        return None            # गेहूँ का season नहीं है
    if day < 20:   return 0.3  # शुरुआती
    if day < 60:   return 0.3 + (day - 20) * (1.15 - 0.3) / 40  # बढ़ती अवस्था
    if day < 110:  return 1.15 # बीच का season
    return 0.4                 # पकने की अवस्था

def simulate(weather, soil_type, threshold=0.5):
    fc, wp = SOILS[soil_type]["fc"], SOILS[soil_type]["wp"]
    taw = fc - wp                      # उपलब्ध पानी
    moisture = fc                      # शुरू में मिट्टी भरी हुई
    rows = []
    for _, r in weather.iterrows():
        date = r["time"]
        kc = kc_for_day(date)
        if kc is None:
            continue
        etc = r["et0_fao_evapotranspiration"] * kc
        rain = r["precipitation_sum"]
        moisture = min(fc, moisture + rain - etc)
        moisture = max(wp, moisture)
        depletion = (fc - moisture) / taw       # 0 = भरी, 1 = सूखी
        irrigate = 1 if depletion > threshold else 0
        water_mm = (fc - moisture) if irrigate else 0
        rows.append({**r.to_dict(), "soil": soil_type, "kc": round(kc, 2),
                     "soil_moisture": round(moisture, 1),
                     "depletion": round(depletion, 2),
                     "irrigate": irrigate, "water_mm": round(water_mm, 1)})
        if irrigate:
            moisture = fc                      # सिंचाई के बाद मिट्टी भर गई
    return pd.DataFrame(rows)

if __name__ == "__main__":
    w = pd.read_csv("data/weather.csv")
    w["time"] = pd.to_datetime(w["time"])
    out = pd.concat([simulate(w, s) for s in SOILS], ignore_index=True)
    out.to_csv("data/dataset.csv", index=False)
    print(out.shape)
    print(out["irrigate"].value_counts())
    print(out.groupby("soil")["water_mm"].sum())