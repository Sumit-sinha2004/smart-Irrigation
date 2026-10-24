# AI-Driven Smart Irrigation Planning Using Weather and Soil Intelligence

B.Tech Major Project (Pre-final Year 2026-27). The system recommends whether to irrigate a wheat field today, and how much water to apply, using live weather and soil moisture. It is compared against fixed-schedule irrigation on seasons the model never saw in training.

## How it works
1. **Data:** daily weather for Greater Noida (2021-2025) from the Open-Meteo API, including reference evapotranspiration (ET0).
2. **Soil moisture simulation:** a daily FAO-56 water balance for wheat on loamy and sandy soil. A day is labelled "irrigate" when more than 50% of the available soil water is used up.
3. **Features:** past 3-day rain and ET0, next 2-day rain, yesterday's soil dryness, crop coefficient, soil type, weather variables.
4. **Models:** Logistic Regression, Random Forest and XGBoost with class weighting (about 4% of days are irrigation days). Split by time: train before 1 June 2024, test after.
5. **Decision logic:** the model prediction plus safety rules (irrigate if critically dry, skip if rain is expected).
6. **Dashboard:** Streamlit app with live forecast, recommendation, and the AI vs fixed-schedule comparison.

## Results
**Model comparison (test set, 404 rows, 15 irrigation days)**

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Logistic Regression | 0.385 | 1.000 | 0.556 |
| Random Forest | 0.706 | 0.800 | 0.750 |
| XGBoost (selected) | 0.737 | 0.933 | 0.824 |

**AI vs fixed schedules (test seasons, 202 days)**

| Soil | Method | Water (mm) | Irrigations | Stress days |
|---|---|---|---|---|
| Loamy | AI (model + rules) | 260 | 5 | 1 |
| Loamy | Fixed every 21 days x 60 mm | 480 | 8 | 7 |
| Sandy | AI (model + rules) | 326 | 9 | 1 |
| Sandy | Fixed every 21 days x 60 mm | 480 | 8 | 24 |

Against the 21-day schedule the AI used 46% (loamy) and 32% (sandy) less water. Results for all schedules are in data/evaluation.csv.

## Project structure
- app.py: Streamlit dashboard
- model.pkl: trained XGBoost model
- src/data.py: download weather data
- src/simulate.py: soil moisture simulation and labels
- src/features.py: feature engineering
- src/train.py: train and compare models
- src/decision.py: model + safety rules
- src/evaluate.py: AI vs fixed schedules, with stress days
- data/: weather, dataset, features, results
- notebooks/: EDA and checks

## How to run
    python -m pip install pandas numpy scikit-learn xgboost matplotlib streamlit requests joblib
    python src/data.py
    python src/simulate.py
    python src/features.py
    python src/train.py
    python src/evaluate.py
    python -m streamlit run app.py

## Limitations
- Soil moisture and irrigation labels are simulated with a rule-based water balance, not measured by sensors. Results are simulation-based validation.
- One location (Greater Noida), one crop (wheat), two soil types.
- Small dataset (1,410 rows, 54 irrigation days); the test set has only 15 irrigation days, so results are indicative.
- The fixed schedules are baselines chosen for comparison, not real farm practice.