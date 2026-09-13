import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier


# ---------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "classification_features.csv"
)

df = pd.read_csv(DATA_PATH)

print(f"Dataset: {df.shape}")


# ---------------------------------------------------------
# LABEL / GROUP
# ---------------------------------------------------------

y = df["label"]

groups = df["flight_id"]

class_names = sorted(y.unique())

label_to_int = {
    label: i
    for i, label in enumerate(class_names)
}

y_encoded = y.map(label_to_int)


# ---------------------------------------------------------
# FEATURE SETS
# ---------------------------------------------------------

metadata = [
    "label",
    "flight_id",
    "window_start",
    "window_end",
]

health_features = [
    "health_score_mean",
    "health_score_std",
    "health_score_min",
    "health_score_delta",
    "health_score_slope",
]

all_features = [
    col for col in df.columns
    if col not in metadata
]

sensor_features = [
    col for col in all_features
    if col not in health_features
]


print(f"\nAll features: {len(all_features)}")
print(f"Sensor features: {len(sensor_features)}")
print(f"Health features: {len(health_features)}")


# ---------------------------------------------------------
# CROSS VALIDATION
# ---------------------------------------------------------

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


def evaluate_feature_set(feature_columns):

    scores = []
    f1_scores = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(
            df[feature_columns],
            y_encoded,
            groups=groups,
        ),
        start=1,
    ):

        X_train = df.iloc[train_idx][feature_columns]
        X_test = df.iloc[test_idx][feature_columns]

        y_train = y_encoded.iloc[train_idx]
        y_test = y_encoded.iloc[test_idx]

        model = XGBClassifier(
            objective="multi:softprob",
            num_class=len(class_names),

            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,

            subsample=0.8,
            colsample_bytree=0.8,

            reg_lambda=1.0,

            eval_metric="mlogloss",

            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )

        scores.append(accuracy)
        f1_scores.append(macro_f1)

        print(
            f"Fold {fold}: "
            f"Accuracy={accuracy:.4f}, "
            f"Macro F1={macro_f1:.4f}"
        )

    return (
        np.mean(scores),
        np.std(scores),
        np.mean(f1_scores),
        np.std(f1_scores),
    )


# ---------------------------------------------------------
# MODEL A — SENSOR + HEALTH SCORE
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("MODEL A — SENSOR + HEALTH SCORE")
print("=" * 60)

a_acc, a_acc_std, a_f1, a_f1_std = evaluate_feature_set(
    all_features
)


# ---------------------------------------------------------
# MODEL B — SENSOR ONLY
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("MODEL B — SENSOR FEATURES ONLY")
print("=" * 60)

b_acc, b_acc_std, b_f1, b_f1_std = evaluate_feature_set(
    sensor_features
)


# ---------------------------------------------------------
# COMPARISON
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("ABLATION RESULTS")
print("=" * 60)

print(
    f"\nModel A — With Health Score:"
    f"\nAccuracy : {a_acc:.4f} ± {a_acc_std:.4f}"
    f"\nMacro F1 : {a_f1:.4f} ± {a_f1_std:.4f}"
)

print(
    f"\nModel B — Without Health Score:"
    f"\nAccuracy : {b_acc:.4f} ± {b_acc_std:.4f}"
    f"\nMacro F1 : {b_f1:.4f} ± {b_f1_std:.4f}"
)

print(
    f"\nAccuracy difference: "
    f"{a_acc - b_acc:+.4f}"
)

print(
    f"Macro F1 difference: "
    f"{a_f1 - b_f1:+.4f}"
)