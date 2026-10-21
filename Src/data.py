import requests
import pandas as pd

def get_weather(lat, lon, start, end):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                 "et0_fao_evapotranspiration,windspeed_10m_max,shortwave_radiation_sum",
        "timezone": "Asia/Kolkata",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return pd.DataFrame(r.json()["daily"])

if __name__ == "__main__":
    df = get_weather(28.47, 77.50, "2021-01-01", "2025-12-31")  # Greater Noida
    df.to_csv("data/weather.csv", index=False)
    print(df.shape)
    print(df.head())