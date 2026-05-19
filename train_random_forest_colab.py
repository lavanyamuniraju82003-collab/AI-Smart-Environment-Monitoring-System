# =====================================================================
# LD7182 — AI-Based Smart Environment Monitoring & Control System
# Random Forest training + TinyML export for ESP32-S3
# Run this notebook in Google Colab end-to-end.
# =====================================================================

# Cell 1 — install dependencies
# !pip install micromlgen scikit-learn pandas matplotlib seaborn -q

# Cell 2 — imports
import warnings, json
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score,
                             recall_score, f1_score)
from micromlgen import port

np.random.seed(42)

# Cell 3 — load the dataset
# (Upload intelligent_indoor_environment_dataset.csv to the Colab workspace.)
df = pd.read_csv("intelligent_indoor_environment_dataset.csv")
print("Rows:", len(df), "Cols:", df.shape[1])
print(df.head())

# Cell 4 — engineer the 3-class status label
def status_label(r):
    t, h, co2, aq = r["room_temperature"], r["room_humidity"], \
                    r["room_CO2"],         r["room_air_quality"]
    score = 0
    if abs(t - 22.5) > 2.0: score += 2
    elif abs(t - 22.5) > 1.2: score += 1
    if h < 38 or h > 47: score += 2
    elif h < 40 or h > 45: score += 1
    if co2 > 425: score += 2
    elif co2 > 410: score += 1
    if aq > 55: score += 2
    elif aq > 51: score += 1
    if score >= 5: return 2     # Critical
    if score >= 2: return 1     # Warning
    return 0                    # Normal

df["status"] = df.apply(status_label, axis=1)
print("Class balance:")
print(df["status"].value_counts())

# Cell 5 — split features / target and train Random Forest
features = ["room_temperature", "room_humidity", "room_CO2",
            "room_air_quality", "lighting_intensity", "room_occupancy"]
X, y = df[features].values, df["status"].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                      random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=120, max_depth=12,
                            min_samples_leaf=2, random_state=42,
                            n_jobs=-1)
rf.fit(Xtr, ytr)

# Cell 6 — evaluate
ypred = rf.predict(Xte)
print(f"Accuracy : {accuracy_score(yte, ypred):.4f}")
print(f"Precision: {precision_score(yte, ypred, average='weighted'):.4f}")
print(f"Recall   : {recall_score(yte, ypred, average='weighted'):.4f}")
print(f"F1       : {f1_score(yte, ypred, average='weighted'):.4f}")
print(f"CV (5)   : {cross_val_score(rf, X, y, cv=5).mean():.4f}")
print(classification_report(yte, ypred,
      target_names=["Normal", "Warning", "Critical"]))

# Cell 7 — confusion matrix
cm = confusion_matrix(yte, ypred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Warning", "Critical"],
            yticklabels=["Normal", "Warning", "Critical"])
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title("Confusion Matrix — Random Forest")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# Cell 8 — feature importance
imp = pd.Series(rf.feature_importances_, index=features).sort_values()
plt.figure(figsize=(7, 4))
imp.plot(kind="barh", color="#1f4e79")
plt.title("Feature Importance — Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()

# Cell 9 — export trained model to C++ header for the ESP32-S3
header = port(rf, classname="RandomForest")
with open("RandomForest.h", "w") as f:
    f.write(header)
print("RandomForest.h saved — drop this file next to your .ino sketch.")
