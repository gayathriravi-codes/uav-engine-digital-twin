"""
check_flight_baseline.py

Does the bucket-(50,100) FP rate correlate with a flight-level baseline
characteristic (raw sensor offset), rather than genuine near-end-of-life
signal? Compares raw per-channel means for high-FP vs low-FP flights,
using the SAME classifier (seed=42, from check_lowrul_classifier.py) and
SAME val split as the prior check.

Val-only. Does not touch test.
Run: python check_flight_baseline.py
"""
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rul import load_all_flights
from train_rul_ensemble import (
    build_windowed_dataset_v5, load_ensemble, predict_rul_ensemble,
    compute_ood_zscore_distance,
)
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
        point_est, std, ood_dist,
        slope(window, CHT_IDX), slope(window, EGT_IDX),
        slope(window, RPM_IDX), slope(window, VIB_IDX),
    ]


def build_matrix(X_set, y_set, models, scaler, dropped_idx_list):
    feats, labels = [], []
    for i in range(len(X_set)):
        result = predict_rul_ensemble(X_set[i], models, scaler, dropped_idx_list, calibrated=False)
        point_est = result["point_estimate_timesteps"]
        std = result["std_timesteps"]
        ood_dist = compute_ood_zscore_distance(X_set[i], scaler)
        feats.append(build_feature_row(X_set[i], point_est, std, ood_dist))
        labels.append(1 if y_set[i] < LOW_RUL_THRESHOLD else 0)
    return np.array(feats), np.array(labels)


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
    val_flight_ids_arr = flight_ids[val_mask]

    models, scaler, dropped_idx_list = load_ensemble()

    print("Refitting seed=42 classifier (matches original check_lowrul_classifier.py run) ...")
    F_train, y_train_bin = build_matrix(X_train, y_train, models, scaler, dropped_idx_list)
    F_val, y_val_bin = build_matrix(X_val, y_val, models, scaler, dropped_idx_list)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(F_train, y_train_bin)
    val_preds = clf.predict(F_val)

    bucket_50_100_mask = (y_val >= 50) & (y_val < 100)

    print("\n=== Per-flight: bucket(50,100) FP rate vs raw baseline sensor means ===")
    print(f"{'flight':30s} {'FP%':>6s} {'n':>4s}  {'cht':>8s} {'egt':>8s} {'rpm':>8s} {'vib':>8s}")

    rows = []
    for fid in sorted(np.unique(val_flight_ids_arr[bucket_50_100_mask])):
        fmask = bucket_50_100_mask & (val_flight_ids_arr == fid)
        n = fmask.sum()
        if n == 0:
            continue
        fp_rate = val_preds[fmask].mean() * 100
        windows = X_val[fmask]  # shape (n, window_len, n_channels)
        cht_mean = windows[:, :, CHT_IDX].mean()
        egt_mean = windows[:, :, EGT_IDX].mean()
        rpm_mean = windows[:, :, RPM_IDX].mean()
        vib_mean = windows[:, :, VIB_IDX].mean()
        rows.append((fid, fp_rate, n, cht_mean, egt_mean, rpm_mean, vib_mean))
        print(f"{fid:30s} {fp_rate:5.0f}% {n:4d}  {cht_mean:8.2f} {egt_mean:8.2f} "
              f"{rpm_mean:8.2f} {vib_mean:8.4f}")

    # Simple correlation check: does FP rate correlate with any raw channel mean?
    print("\n=== Correlation of FP rate with each raw channel's flight-level mean ===")
    fp_rates = np.array([r[1] for r in rows])
    for idx, name in enumerate(["cht", "egt", "rpm", "vib"], start=3):
        vals = np.array([r[idx] for r in rows])
        if np.std(vals) == 0 or np.std(fp_rates) == 0:
            print(f"  {name}: n/a (zero variance)")
            continue
        corr = np.corrcoef(fp_rates, vals)[0, 1]
        print(f"  {name}: r={corr:+.3f}")

    print("\nLook for: does FP% track a specific channel's mean level (a real baseline-offset")
    print("story), or is there no clear correlation (meaning the FP variation isn't explained")
    print("by simple per-flight sensor offset, and needs a different explanation)?")


if __name__ == "__main__":
    main()