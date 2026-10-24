import joblib
import pandas as pd

from decision import recommend
from simulate import SOILS

STRESS_LEVEL = 0.5          # depletion above this = crop water stress
CUTOFF = pd.Timestamp("2024-06-01")   # same split as train.py


def run_season(season, soil, policy):
    """Simulate one season day by day.
    policy(i, row, depletion) returns the mm of water to apply that day."""
    fc, wp = SOILS[soil]["fc"], SOILS[soil]["wp"]
    taw = fc - wp
    moisture = fc
    applied = excess = 0.0
    events = stress_days = 0
    for i, (_, r) in enumerate(season.iterrows()):
        depletion = (fc - moisture) / taw
        irr = policy(i, r, depletion)
        moisture += r["precipitation_sum"] - r["et0_fao_evapotranspiration"] * r["kc"] + irr
        excess += max(0.0, moisture - fc)
        moisture = max(wp, min(fc, moisture))
        applied += irr
        if irr > 0:
            events += 1
        if (fc - moisture) / taw > STRESS_LEVEL:
            stress_days += 1
    return applied, excess, events, stress_days


def make_ai_policy(model, soil):
    def policy(i, r, depletion):
        w = {
            "tmax": r["temperature_2m_max"], "tmin": r["temperature_2m_min"],
            "rain": r["precipitation_sum"], "et0": r["et0_fao_evapotranspiration"],
            "wind": r["windspeed_10m_max"], "rad": r["shortwave_radiation_sum"],
            "kc": r["kc"], "rain_last3": r["rain_last3"], "et0_last3": r["et0_last3"],
            "rain_next2": r["rain_next2"], "prev_depletion": depletion,
        }
        msg, water = recommend(model, soil, w)
        return water if msg.startswith("Irrigate") else 0
    return policy


def make_fixed_policy(interval, depth):
    def policy(i, r, depletion):
        return depth if (i + 1) % interval == 0 else 0
    return policy


if __name__ == "__main__":
    model = joblib.load("model.pkl")
    df = pd.read_csv("data/features.csv")
    df["time"] = pd.to_datetime(df["time"])
    df["season"] = df["time"].apply(lambda d: d.year if d.month >= 11 else d.year - 1)

    rows = []
    for soil in SOILS:
        for yr, season in df[df["soil"] == soil].groupby("season"):
            season = season.sort_values("time")
            split = "test" if season["time"].min() >= CUTOFF else "train"
            policies = {
                "AI (model + rules)": make_ai_policy(model, soil),
                "Fixed every 10d x 50mm": make_fixed_policy(10, 50),
                "Fixed every 15d x 60mm": make_fixed_policy(15, 60),
                "Fixed every 21d x 60mm": make_fixed_policy(21, 60),
            }
            for name, pol in policies.items():
                applied, excess, events, stress = run_season(season, soil, pol)
                rows.append({
                    "split": split, "season": yr, "days": len(season), "soil": soil,
                    "method": name, "water_mm": applied, "excess_mm": excess,
                    "irrigations": events, "stress_days": stress,
                })

    res = pd.DataFrame(rows).round(0)
    res.to_csv("data/evaluation.csv", index=False)

    summary = res.groupby(["split", "soil", "method"])[
        ["water_mm", "excess_mm", "irrigations", "stress_days"]].sum()
    print(summary.to_string())