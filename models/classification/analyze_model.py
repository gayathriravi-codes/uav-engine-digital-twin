import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier


# ---------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "classification"
    / "xgboost_fault_classifier.json"
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "classification_features.csv"
)


# ---------------------------------------------------------
# LOAD
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)

DROP_COLUMNS = [
    "label",
    "flight_id",
    "window_start",
    "window_end",
]

X = df.drop(columns=DROP_COLUMNS)

model = XGBClassifier()
model.load_model(MODEL_PATH)


# ---------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_,
})

importance = importance.sort_values(
    "importance",
    ascending=False,
)


print("\nTop 20 Features:")
print(
    importance.head(20).to_string(
        index=False
    )
)


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

output_path = (
    PROJECT_ROOT
    / "models"
    / "classification"
    / "feature_importance.csv"
)

importance.to_csv(
    output_path,
    index=False
)

print(
    f"\nSaved feature importance to:\n"
    f"{output_path}"
)


# ---------------------------------------------------------
# PLOT
# ---------------------------------------------------------

top = importance.head(20).sort_values(
    "importance"
)

plt.figure(figsize=(10, 8))

plt.barh(
    top["feature"],
    top["importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("XGBoost Fault Classifier - Top 20 Features")

plt.tight_layout()

plot_path = (
    PROJECT_ROOT
    / "models"
    / "classification"
    / "feature_importance.png"
)

plt.savefig(plot_path, dpi=150)

print(
    f"Saved plot to:\n"
    f"{plot_path}"
)

plt.show()