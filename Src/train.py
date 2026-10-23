import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from features import FEATURES

df = pd.read_csv("data/features.csv")
df["time"] = pd.to_datetime(df["time"])

# Time-based split: train on the past, test on the future
cutoff = pd.Timestamp("2024-06-01")
train, test = df[df["time"] < cutoff], df[df["time"] >= cutoff]
X_train, y_train = train[FEATURES], train["irrigate"]
X_test, y_test = test[FEATURES], test["irrigate"]
print("Train rows:", len(train), "| irrigation days:", y_train.sum())
print("Test rows:", len(test), "| irrigation days:", y_test.sum())

# Handle class imbalance (only ~4% of days are irrigation days)
pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

models = {
    "LogisticRegression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=200, scale_pos_weight=pos_weight,
        eval_metric="logloss", random_state=42),
}

results = []
for name, m in models.items():
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    results.append({
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
    })
    print("\n", name)
    print(confusion_matrix(y_test, pred))

res = pd.DataFrame(results).round(3)
print("\n", res.to_string(index=False))
res.to_csv("data/model_results.csv", index=False)

# Save the best model (by F1)
best_name = res.sort_values("f1", ascending=False).iloc[0]["model"]
joblib.dump(models[best_name], "model.pkl")
print("\nBest model:", best_name, "(saved as model.pkl)")

# Feature importance plot (Random Forest)
rf = models["RandomForest"]
imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
imp.plot(kind="barh", figsize=(8, 5), title="Feature importance (Random Forest)")
plt.tight_layout()
plt.savefig("data/feature_importance.png")
print("Saved data/feature_importance.png")