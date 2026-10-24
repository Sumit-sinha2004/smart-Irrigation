import sys
sys.path.insert(0, "src")

import joblib
import pandas as pd
import requests
import streamlit as st

from decision import recommend
from simulate import SOILS

st.set_page_config(page_title="Smart Irrigation Planner", layout="wide")

LAT, LON = 28.47, 77.50  # Greater Noida


def kc_from_day(day):
    if day < 20:
        return 0.3
    if day < 60:
        return 0.3 + (day - 20) * (1.15 - 0.3) / 40
    if day < 110:
        return 1.15
    return 0.4


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


@st.cache_data(ttl=3600)
def get_forecast():
    params = {
        "latitude": LAT, "longitude": LON,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                 "et0_fao_evapotranspiration,windspeed_10m_max,shortwave_radiation_sum",
        "past_days": 3, "forecast_days": 3, "timezone": "Asia/Kolkata",
    }
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=30)
    r.raise_for_status()
    return pd.DataFrame(r.json()["daily"])


@st.cache_data
def load_dataset():
    df = pd.read_csv("data/dataset.csv")
    df["time"] = pd.to_datetime(df["time"])
    df["season"] = df["time"].apply(lambda d: d.year if d.month >= 11 else d.year - 1)
    return df


def fixed_schedule(season, soil, interval, depth):
    fc, wp = SOILS[soil]["fc"], SOILS[soil]["wp"]
    moisture, applied, events = fc, 0.0, 0
    for i, (_, r) in enumerate(season.iterrows()):
        irr = depth if (i + 1) % interval == 0 else 0
        moisture += r["precipitation_sum"] - r["et0_fao_evapotranspiration"] * r["kc"] + irr
        moisture = max(wp, min(fc, moisture))
        applied += irr
        events += 1 if irr else 0
    return applied, events


model = load_model()

st.title("AI-Driven Smart Irrigation Planner")
st.caption("Greater Noida | Wheat | Live weather from Open-Meteo")

with st.sidebar:
    st.header("Field inputs")
    soil = st.selectbox("Soil type", list(SOILS.keys()))
    day = st.slider("Days since sowing", 0, 140, 60)
    depletion = st.slider("Yesterday's soil dryness (0 = full, 1 = dry)", 0.0, 1.0, 0.3, 0.05)

tab1, tab2 = st.tabs(["Today's recommendation", "Water saved vs fixed schedule"])

with tab1:
    try:
        fc_df = get_forecast()
    except Exception as e:
        st.error(f"Could not load the weather forecast: {e}")
        st.stop()

    today = fc_df.iloc[3]
    w = {
        "tmax": today["temperature_2m_max"], "tmin": today["temperature_2m_min"],
        "rain": today["precipitation_sum"], "et0": today["et0_fao_evapotranspiration"],
        "wind": today["windspeed_10m_max"], "rad": today["shortwave_radiation_sum"],
        "kc": kc_from_day(day),
        "rain_last3": fc_df["precipitation_sum"].iloc[0:3].sum(),
        "et0_last3": fc_df["et0_fao_evapotranspiration"].iloc[0:3].mean(),
        "rain_next2": fc_df["precipitation_sum"].iloc[4:6].sum(),
        "prev_depletion": depletion,
    }

    msg, water = recommend(model, soil, w)
    if msg.startswith("Irrigate"):
        st.warning(f"{msg}: apply about {water} mm")
    else:
        st.success(msg)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max temp (C)", f"{w['tmax']:.1f}")
    c2.metric("Rain today (mm)", f"{w['rain']:.1f}")
    c3.metric("Rain next 2 days (mm)", f"{w['rain_next2']:.1f}")
    c4.metric("ET0 today (mm)", f"{w['et0']:.1f}")

    st.subheader("Weather used (3 past days, today, 2 forecast days)")
    st.dataframe(fc_df, use_container_width=True)

with tab2:
    data = load_dataset()
    c1, c2, c3 = st.columns(3)
    season_year = c1.selectbox("Season (sowing year)", sorted(data["season"].unique()))
    interval = c2.slider("Fixed schedule: irrigate every N days", 5, 20, 10)
    depth = c3.slider("Fixed schedule: water per irrigation (mm)", 20, 100, 50)

    season = data[(data["season"] == season_year) & (data["soil"] == soil)].sort_values("time")
    ai_water = season["water_mm"].sum()
    ai_events = int(season["irrigate"].sum())
    fx_water, fx_events = fixed_schedule(season, soil, interval, depth)

    m1, m2, m3 = st.columns(3)
    m1.metric("AI-based water (mm)", f"{ai_water:.0f}", f"{ai_events} irrigations")
    m2.metric("Fixed-schedule water (mm)", f"{fx_water:.0f}", f"{fx_events} irrigations")
    saved = (fx_water - ai_water) / fx_water * 100 if fx_water else 0
    m3.metric("Water saved", f"{saved:.0f}%")

    st.bar_chart(pd.DataFrame({"Water (mm)": [ai_water, fx_water]},
                              index=["AI-based", "Fixed schedule"]))
    st.caption("A fixed schedule that applies too little water would also look efficient, "
               "so crop stress days will be added in the testing step.")