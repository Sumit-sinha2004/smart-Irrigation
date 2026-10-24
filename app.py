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
AI = "AI (model + rules)"
METHODS = [AI, "Fixed every 10d x 50mm", "Fixed every 15d x 60mm", "Fixed every 21d x 60mm"]


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
def load_eval():
    return pd.read_csv("data/evaluation.csv")


model = load_model()

st.title("AI-Driven Smart Irrigation Planner")
st.caption("Greater Noida | Wheat | Live weather from Open-Meteo")

with st.sidebar:
    st.header("Field inputs")
    soil = st.selectbox("Soil type", list(SOILS.keys()))
    day = st.slider("Days since sowing", 0, 140, 60)
    depletion = st.slider("Yesterday's soil dryness (0 = full, 1 = dry)", 0.0, 1.0, 0.3, 0.05)

tab1, tab2 = st.tabs(["Today's recommendation", "AI vs fixed schedules"])

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
    st.dataframe(fc_df, width="stretch")

with tab2:
    ev = load_eval()
    split = st.radio("Seasons used", ["test", "train"], horizontal=True,
                     help="test = seasons the model never saw in training (2024 season "
                          "and Nov-Dec 2025). train = earlier seasons.")
    sub = ev[(ev["split"] == split) & (ev["soil"] == soil)]
    tot = sub.groupby("method")[["water_mm", "excess_mm", "irrigations", "stress_days"]].sum()
    tot = tot.reindex(METHODS)

    baseline = st.selectbox("Compare the AI against", METHODS[1:], index=2)
    ai, fx = tot.loc[AI], tot.loc[baseline]
    saved = (fx["water_mm"] - ai["water_mm"]) / fx["water_mm"] * 100

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AI water (mm)", f"{ai['water_mm']:.0f}", f"{ai['irrigations']:.0f} irrigations",
              delta_color="off")
    m2.metric("Fixed-schedule water (mm)", f"{fx['water_mm']:.0f}",
              f"{fx['irrigations']:.0f} irrigations", delta_color="off")
    m3.metric("Water saved by AI", f"{saved:.0f}%")
    m4.metric("Stress days: AI vs fixed",
              f"{ai['stress_days']:.0f} vs {fx['stress_days']:.0f}")

    st.subheader(f"Water applied (mm), {soil} soil, {split} seasons")
    st.bar_chart(tot["water_mm"])

    st.subheader("Full comparison")
    st.dataframe(tot.rename(columns={
        "water_mm": "Water (mm)", "excess_mm": "Excess above field capacity (mm)",
        "irrigations": "Irrigations", "stress_days": "Crop stress days"}), width="stretch")

    st.caption("Simulation-based validation: stress days are days when more than 50% of the "
               "available soil water is used up. Fixed schedules are baselines chosen for "
               "comparison, not real farm practice.")