"""
ablation_no_correlation.py -- same leak-fixed training as train_xgboost_v2.py,
but with the corr_rpm_fuel_flow / corr_egt_cht / *_collapse columns DROPPED.

Purpose: the egt<->cht baseline correlation came out near-zero (0.026), which
means the correlation-collapse feature's real contribution to v2's 89.95%
result was not yet established -- it could be doing nothing. This is the
only honest way to know before claiming in the deck that the feature helped.

Compare this script's final accuracy/F1 to train_xgboost_v2.py's:
  - If accuracy drops meaningfully without the correlation features ->
    they're pulling real weight, the "physics-informed feature" claim holds.
  - If accuracy is unchanged (or even better) without them -> say so plainly;
    it's still fine to keep the feature as a legitimate idea, but claim it
    honestly as "did not measurably help in this dataset" rather than
    "improved performance."

Run from repo root: python models/classification/ablation_no_correlation.py
(requires classification_features_v2.csv to already exist)
"""
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
)

from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "classification_features_v2.csv"
df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset: {df.shape}")

df["base_id"] = df["flight_id"].str.rsplit("_", n=1).str[-1]

CORR_COLUMNS = [
    "corr_rpm_fuel_flow", "corr_rpm_fuel_flow_collapse",
    "corr_egt_cht", "corr_egt_cht_collapse",
]
DROP_COLUMNS = ["label", "flight_id", "window_start", "window_end", "base_id"] + CORR_COLUMNS

missing = [c for c in CORR_COLUMNS if c not in df.columns]
if missing:
    raise ValueError(f"Expected correlation columns not found: {missing} -- "
                      f"did you run feature_extraction_v2.py?")

X = df.drop(columns=DROP_COLUMNS)
y = df["label"]
groups = df["base_id"]

print(f"\nFeature count WITHOUT correlation columns: {X.shape[1]} "
      f"(vs. {X.shape[1] + len(CORR_COLUMNS)} with them)")

class_names = sorted(y.unique())
label_to_int = {label: i for i, label in enumerate(class_names)}
y_encoded = y.map(label_to_int)

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

fold_results = []
all_true, all_pred = [], []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y_encoded, groups=groups), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_encoded.iloc[train_idx], y_encoded.iloc[test_idx]
    groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]

    overlap = set(groups_train.unique()) & set(groups_test.unique())
    assert not overlap, f"LEAKAGE: base trajectories {overlap} appear in both train and test!"

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

    print(f"Fold {fold}: acc={accuracy:.4f} prec={precision:.4f} rec={recall:.4f} f1={f1:.4f}")

    fold_results.append([accuracy, precision, recall, f1])
    all_true.extend(y_test)
    all_pred.extend(predictions)

results = np.array(fold_results)
print("\n" + "=" * 60)
print("ABLATION SUMMARY -- WITHOUT correlation-collapse features")
print("=" * 60)
print(f"Accuracy : {results[:, 0].mean():.4f} +/- {results[:, 0].std():.4f}")
print(f"Precision: {results[:, 1].mean():.4f} +/- {results[:, 1].std():.4f}")
print(f"Recall   : {results[:, 2].mean():.4f} +/- {results[:, 2].std():.4f}")
print(f"F1 Score : {results[:, 3].mean():.4f} +/- {results[:, 3].std():.4f}")
print("\nCompare directly to train_xgboost_v2.py's WITH-correlation result:")
print("  Accuracy : 0.8995 +/- 0.0132")
print("  Precision: 0.9046 +/- 0.0107")
print("  Recall   : 0.8995 +/- 0.0132")
print("  F1 Score : 0.8956 +/- 0.0144")

print("\n" + "=" * 60)
print("PER-CLASS REPORT -- WITHOUT correlation-collapse features")
print("=" * 60)
print(classification_report(all_true, all_pred, target_names=class_names, zero_division=0))
print("\nCompare especially: cooling_degradation and overheat rows (the two")
print("faults the correlation feature was designed to help distinguish).")
