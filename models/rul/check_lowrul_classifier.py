"""
check_lowrul_classifier.py  (v2 - fixed key names + real OOD call signature)
"""
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rul import load_all_flights
from train_rul_ensemble import (
    build_windowed_dataset_v5, load_ensemble, predict_rul_ensemble,
    compute_ood_zscore_distance,
)
from calibrate_rul import load_calibration_params
from schema import SENSOR_FIELDS

CHT_IDX = SENSOR_FIELDS.index("cht")
EGT_IDX = SENSOR_FIELDS.index("egt")
RPM_IDX = SENSOR_FIELDS.index("rpm")
VIB_IDX = SENSOR_FIELDS.index("vibration")

LOW_RUL_THRESHOLD = 50.0


def slope(window, idx, n=10):
    series = window[:, idx]
    tail = series[-n:]
    return (tail[-1] - tail[0]) / len(tail)


def build_feature_row(window, point_est, std, ood_dist):
    return [
        point_est,
        std,
        ood_dist,
        slope(window, CHT_IDX),
        slope(window, EGT_IDX),
        slope(window, RPM_IDX),
        slope(window, VIB_IDX),
    ]


def main():
    print("Loading flights and building windowed dataset ...")
    flights = load_all_flights()
    X, y, flight_ids = build_windowed_dataset_v5(flights)

    unique_flights = np.unique(flight_ids)
    train_flights, temp_flights = train_test_split(unique_flights, test_size=0.3, random_state=42)
    val_flights, test_flights = train_test_split(temp_flights, test_size=0.5, random_state=42)

    train_mask = np.isin(flight_ids, train_flights)
    val_mask = np.isin(flight_ids, val_flights)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    print(f"Train windows: {len(X_train)}  Val windows: {len(X_val)}")

    models, scaler, dropped_idx_list = load_ensemble()
    load_calibration_params()

    def build_matrix(X_set, y_set):
        feats, labels, point_ests = [], [], []
        for i in range(len(X_set)):
            result = predict_rul_ensemble(X_set[i], models, scaler, dropped_idx_list, calibrated=False)
            point_est = result["point_estimate_timesteps"]
            std = result["std_timesteps"]
            ood_dist = compute_ood_zscore_distance(X_set[i], scaler)
            feats.append(build_feature_row(X_set[i], point_est, std, ood_dist))
            labels.append(1 if y_set[i] < LOW_RUL_THRESHOLD else 0)
            point_ests.append(point_est)
        return np.array(feats), np.array(labels), np.array(point_ests)

    print("\nBuilding feature matrices (calls predict_rul_ensemble + compute_ood_zscore_distance per window) ...")
    F_train, y_train_bin, _ = build_matrix(X_train, y_train)
    F_val, y_val_bin, val_point_ests = build_matrix(X_val, y_val)

    print(f"\nTrain: {y_train_bin.sum()} low-RUL / {len(y_train_bin)} total "
          f"({100*y_train_bin.mean():.1f}%)")
    print(f"Val:   {y_val_bin.sum()} low-RUL / {len(y_val_bin)} total "
          f"({100*y_val_bin.mean():.1f}%)")

    print("\n=== Training logistic regression (class_weight='balanced') ===")
    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(F_train, y_train_bin)

    val_probs = clf.predict_proba(F_val)[:, 1]
    val_preds = clf.predict(F_val)

    acc = accuracy_score(y_val_bin, val_preds)
    auc = roc_auc_score(y_val_bin, val_probs)
    prec = precision_score(y_val_bin, val_preds)
    rec = recall_score(y_val_bin, val_preds)

    print(f"  Accuracy: {acc:.3f}")
    print(f"  AUC:      {auc:.3f}")
    print(f"  Precision (low-RUL class): {prec:.3f}")
    print(f"  Recall (low-RUL class):    {rec:.3f}")

    print("\n=== Recall specifically on the known-failing subset (point_est overshoot > 20) ===")
    overshoot_mask = (y_val_bin == 1) & ((val_point_ests - y_val) > 20)
    n_failing = overshoot_mask.sum()
    caught = np.sum((val_preds == 1) & overshoot_mask)
    print(f"  Known failing windows in this val split: {n_failing}")
    if n_failing:
        print(f"  Of those, classifier flagged as low-RUL: {caught} / {n_failing}  "
              f"({100*caught/n_failing:.1f}%)")
    else:
        print("  (none found in this split)")

    print("\n=== False positive rate by bucket (bucket >= 50, should NOT be flagged) ===")
    for lo, hi in [(50, 100), (100, 200), (200, 350)]:
        bmask = (y_val >= lo) & (y_val < hi)
        if bmask.sum() == 0:
            continue
        fp_rate = val_preds[bmask].mean()
        print(f"  bucket ({lo},{hi}): n={bmask.sum():4d}  flagged-as-low-RUL={val_preds[bmask].sum():4d}  "
              f"FP rate={100*fp_rate:.1f}%")

    print("\n=== Logistic regression coefficients (standardized scale) ===")
    feat_names = ["point_est", "std", "ood_dist", "cht_slope", "egt_slope", "rpm_slope", "vib_slope"]
    for name, coef in zip(feat_names, clf.coef_[0]):
        print(f"  {name:12s}: {coef:+.4f}")

    print("\nSTOP-OR-CONTINUE: if recall on the known-failing subset is low, or false positive")
    print("rate on bucket(50,100)+ is high, this track is a dead end -- report back rather")
    print("than building the wiring. Don't run run_coverage_check.py.")


if __name__ == "__main__":
    main()