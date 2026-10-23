import pandas as pd

FEATURES = [
    "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
    "et0_fao_evapotranspiration", "windspeed_10m_max", "shortwave_radiation_sum",
    "kc", "soil_is_sandy",
    "rain_last3", "et0_last3", "rain_next2", "prev_depletion",
]


def add_soil_features(g, soil):
    g = g.sort_values("time").copy()
    g["rain_last3"] = g["precipitation_sum"].rolling(3, min_periods=1).sum().shift(1).fillna(0)
    g["et0_last3"] = g["et0_fao_evapotranspiration"].rolling(3, min_periods=1).mean().shift(1).fillna(0)
    g["rain_next2"] = g["precipitation_sum"][::-1].rolling(2, min_periods=1).sum()[::-1].shift(-1).fillna(0)
    g["prev_depletion"] = g["depletion"].where(g["irrigate"] == 0, 0).shift(1).fillna(0)
    g["soil_is_sandy"] = 1 if soil == "sandy" else 0
    return g


def build_features(df):
    parts = [add_soil_features(g, soil) for soil, g in df.groupby("soil")]
    return pd.concat(parts).sort_values(["time", "soil"]).reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_csv("data/dataset.csv")
    df["time"] = pd.to_datetime(df["time"])
    out = build_features(df)
    out.to_csv("data/features.csv", index=False)
    print(out.shape)
    print(out[FEATURES + ["irrigate"]].head())