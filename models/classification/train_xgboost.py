import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from xgboost import XGBClassifier


# ---------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# LOAD FEATURE DATASET
# ---------------------------------------------------------

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "classification_features.csv"
)

df = pd.read_csv(DATA_PATH)

print(f"Loaded dataset: {df.shape}")


# ---------------------------------------------------------
# SEPARATE FEATURES / LABEL / GROUP
# ---------------------------------------------------------

DROP_COLUMNS = [
    "label",
    "flight_id",
    "window_start",
    "window_end",
]

X = df.drop(columns=DROP_COLUMNS)
y = df["label"]
groups = df["flight_id"]


# Convert labels to integer classes
class_names = sorted(y.unique())

label_to_int = {
    label: i
    for i, label in enumerate(class_names)
}

int_to_label = {
    i: label
    for label, i in label_to_int.items()
}

y_encoded = y.map(label_to_int)


print("\nClasses:")
for i, name in int_to_label.items():
    print(f"{i}: {name}")


# ---------------------------------------------------------
# GROUP-AWARE CROSS VALIDATION
# ---------------------------------------------------------

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


fold_results = []

all_true = []
all_pred = []


# ---------------------------------------------------------
# TRAIN / VALIDATE
# ---------------------------------------------------------

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y_encoded, groups=groups),
    start=1,
):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y_encoded.iloc[train_idx]
    y_test = y_encoded.iloc[test_idx]

    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]

    print("\n" + "=" * 60)
    print(f"FOLD {fold}")
    print("=" * 60)

    print(f"Training windows: {len(X_train)}")
    print(f"Validation windows: {len(X_test)}")

    print(
        f"Training flights: "
        f"{groups_train.nunique()}"
    )

    print(
        f"Validation flights: "
        f"{groups_test.nunique()}"
    )

    # -----------------------------------------------------
    # XGBOOST
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    fold_results.append(
        [accuracy, precision, recall, f1]
    )

    all_true.extend(y_test)
    all_pred.extend(predictions)


# ---------------------------------------------------------
# CROSS-VALIDATION SUMMARY
# ---------------------------------------------------------

results = np.array(fold_results)

print("\n" + "=" * 60)
print("5-FOLD CROSS-VALIDATION SUMMARY")
print("=" * 60)

print(
    f"Accuracy : "
    f"{results[:, 0].mean():.4f} ± "
    f"{results[:, 0].std():.4f}"
)

print(
    f"Precision: "
    f"{results[:, 1].mean():.4f} ± "
    f"{results[:, 1].std():.4f}"
)

print(
    f"Recall   : "
    f"{results[:, 2].mean():.4f} ± "
    f"{results[:, 2].std():.4f}"
)

print(
    f"F1 Score : "
    f"{results[:, 3].mean():.4f} ± "
    f"{results[:, 3].std():.4f}"
)


# ---------------------------------------------------------
# OVERALL CLASSIFICATION REPORT
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("OVERALL CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        all_true,
        all_pred,
        target_names=class_names,
        zero_division=0,
    )
)


# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------

cm = confusion_matrix(
    all_true,
    all_pred,
)

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names,
)

print("\nConfusion Matrix:")
print(cm_df)


# ---------------------------------------------------------
# TRAIN FINAL MODEL ON ALL DATA
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("TRAINING FINAL MODEL")
print("=" * 60)

final_model = XGBClassifier(
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

final_model.fit(
    X,
    y_encoded,
)


# ---------------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------------

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "classification"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_PATH = (
    MODEL_DIR
    / "xgboost_fault_classifier.json"
)

final_model.save_model(
    MODEL_PATH
)

print(
    f"\nFinal model saved to:\n"
    f"{MODEL_PATH}"
)