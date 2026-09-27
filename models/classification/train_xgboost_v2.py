"""
train_xgboost_v2.py -- same leak-fixed base_id-grouped StratifiedGroupKFold
approach as train_xgboost_fixed.py, trained on classification_features_v2.csv
(7 core sensors + 2 new sensors + cross-sensor correlation-collapse features,
9 fault classes + none).

Saves to a SEPARATE model file (xgboost_fault_classifier_v2.json) --
xgboost_fault_classifier_fixed.json and its 91.1% result are untouched,
so this is purely additive: if v2 underperforms or something looks wrong,
the original result is still there as a fallback.
"""
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "classification_features_v2.csv"

df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset: {df.shape}")

# ---------------------------------------------------------
# SAME LEAKAGE FIX AS train_xgboost_fixed.py
# ---------------------------------------------------------
df["base_id"] = df["flight_id"].str.rsplit("_", n=1).str[-1]

print(f"\nBase trajectories: {df['base_id'].nunique()}")
print(df.groupby("base_id")["flight_id"].nunique().to_string())

DROP_COLUMNS = ["label", "flight_id", "window_start", "window_end", "base_id"]
X = df.drop(columns=DROP_COLUMNS)
y = df["label"]
groups = df["base_id"]

class_names = sorted(y.unique())
label_to_int = {label: i for i, label in enumerate(class_names)}
int_to_label = {i: label for label, i in label_to_int.items()}
y_encoded = y.map(label_to_int)

print("\nClasses:")
for i, name in int_to_label.items():
    print(f"{i}: {name}")

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

fold_results = []
all_true, all_pred = [], []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y_encoded, groups=groups), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_encoded.iloc[train_idx], y_encoded.iloc[test_idx]
    groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]

    overlap = set(groups_train.unique()) & set(groups_test.unique())
    assert not overlap, f"LEAKAGE: base trajectories {overlap} appear in both train and test!"

    print("\n" + "=" * 60)
    print(f"FOLD {fold}")
    print("=" * 60)
    print(f"Training windows: {len(X_train)} | Validation windows: {len(X_test)}")
    print(f"Training base trajectories: {groups_train.nunique()} | Validation: {groups_test.nunique()}")

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
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
    recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
    f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    fold_results.append([accuracy, precision, recall, f1])
    all_true.extend(y_test)
    all_pred.extend(predictions)

results = np.array(fold_results)
print("\n" + "=" * 60)
print("5-FOLD CROSS-VALIDATION SUMMARY (v2 -- extended sensors + correlation features)")
print("=" * 60)
print(f"Accuracy : {results[:, 0].mean():.4f} +/- {results[:, 0].std():.4f}")
print(f"Precision: {results[:, 1].mean():.4f} +/- {results[:, 1].std():.4f}")
print(f"Recall   : {results[:, 2].mean():.4f} +/- {results[:, 2].std():.4f}")
print(f"F1 Score : {results[:, 3].mean():.4f} +/- {results[:, 3].std():.4f}")

print("\n" + "=" * 60)
print("OVERALL CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(all_true, all_pred, target_names=class_names, zero_division=0))

cm = confusion_matrix(all_true, all_pred)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
print("\nConfusion Matrix:")
print(cm_df)

# ---------------------------------------------------------
# SEPARABILITY CHECK: misfire vs injector_abnormality
# ---------------------------------------------------------
# The whole point of designing injector_abnormality's fuel_flow-pulsing
# signature to differ from misfire's RPM-drop signature was so a real
# classifier could tell them apart, not just a sanity-check plot. Print
# this pair's confusion explicitly so it's not buried in the full matrix.
if "misfire" in cm_df.index and "injector_abnormality" in cm_df.index:
    print("\n" + "=" * 60)
    print("SEPARABILITY CHECK: misfire vs injector_abnormality")
    print("=" * 60)
    print(cm_df.loc[["misfire", "injector_abnormality"], ["misfire", "injector_abnormality"]])

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
final_model.fit(X, y_encoded)

MODEL_DIR = PROJECT_ROOT / "models" / "classification"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "xgboost_fault_classifier_v2.json"
final_model.save_model(MODEL_PATH)

print(f"\nFinal model saved to:\n{MODEL_PATH}")
print("\n(xgboost_fault_classifier_fixed.json is untouched -- original 91.1% result still available as fallback)")
