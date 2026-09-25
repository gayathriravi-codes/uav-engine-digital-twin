"""
check_classifier_stability.py

Bug 2, Steps 5-6: is the low-RUL classifier's result (26/28 recall on known
failing windows, 38.1% FP rate in bucket(50,100)) stable across seeds, and
is either number driven by a collapse on 1-2 flights rather than a general
signal?

Multi-seed: bootstrap-resample TRAIN flights (bootstrap_by_flight, confirmed
to exist in train_rul_ensemble.py) across several seeds, refit the same
logistic regression, recheck recall/FP on the SAME val set each time.

Per-flight: break down both the failing-window catch rate and the
bucket-(50,100) false positives by flight_id, so a result that looks fine
in aggregate can't hide a collapse on a couple of flights -- this is the
exact failure mode that made the reweighting experiment look better than
it was.

Val-only. Does not touch test.
Run: python check_classifier_stability.py
"""
import os
import sys
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rul import load_all_flights
from train_rul_ensemble import (
    build_windowed_dataset_v5, load_ensemble, predict_rul_ensemble,
    compute_ood_zscore_distance, bootstrap_by_flight,
)
from schema import SENSOR_FIELDS

CHT_IDX = SENSOR_FIELDS.index("cht")
EGT_IDX = SENSOR_FIELDS.index("egt")
RPM_IDX = SENSOR_FIELDS.index("rpm")
VIB_IDX = SENSOR_FIELDS.index("vibration")

LOW_RUL_THRESHOLD = 50.0
SEEDS = [42, 7, 123, 2024]


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
    train_flight_ids = flight_ids[train_mask]
    val_flight_ids_arr = flight_ids[val_mask]

    models, scaler, dropped_idx_list = load_ensemble()

    print("Building VAL feature matrix once (val set doesn't change across seeds) ...")
    F_val, y_val_bin, val_point_ests = build_matrix(X_val, y_val, models, scaler, dropped_idx_list)
    overshoot_mask = (y_val_bin == 1) & ((val_point_ests - y_val) > 20)
    n_failing = overshoot_mask.sum()
    bucket_50_100_mask = (y_val >= 50) & (y_val < 100)

    print(f"Known failing windows in val: {n_failing}")
    print(f"Bucket (50,100) windows in val: {bucket_50_100_mask.sum()}\n")

    print("=" * 78)
    print("MULTI-SEED STABILITY (bootstrap-resample train flights, refit, same val set)")
    print("=" * 78)

    seed_results = []
    for seed in SEEDS:
        X_train_boot, y_train_boot = bootstrap_by_flight(X_train, y_train, train_flight_ids, seed)
        # Rebuild features for the bootstrap sample (point_est/std/ood are deterministic
        # given the window, so this recomputes per-window features on the resampled set)
        F_train_boot, y_train_bin_boot, _ = build_matrix(
            X_train_boot, y_train_boot, models, scaler, dropped_idx_list
        )

        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
        clf.fit(F_train_boot, y_train_bin_boot)

        val_preds = clf.predict(F_val)
        caught = np.sum((val_preds == 1) & overshoot_mask)
        recall_failing = caught / n_failing if n_failing else float("nan")
        fp_rate_50_100 = val_preds[bucket_50_100_mask].mean() if bucket_50_100_mask.sum() else float("nan")

        seed_results.append({
            "seed": seed, "recall_failing": recall_failing, "caught": caught,
            "fp_rate_50_100": fp_rate_50_100, "preds": val_preds,
        })
        print(f"  seed={seed:5d}  failing-recall={100*recall_failing:5.1f}% ({caught}/{n_failing})  "
              f"bucket(50,100) FP rate={100*fp_rate_50_100:5.1f}%")

    recalls = [r["recall_failing"] for r in seed_results]
    fps = [r["fp_rate_50_100"] for r in seed_results]
    print(f"\n  Recall across seeds:  min={100*min(recalls):.1f}%  max={100*max(recalls):.1f}%  "
          f"spread={100*(max(recalls)-min(recalls)):.1f}pp")
    print(f"  FP rate across seeds: min={100*min(fps):.1f}%  max={100*max(fps):.1f}%  "
          f"spread={100*(max(fps)-min(fps)):.1f}pp")

    print("\n" + "=" * 78)
    print("PER-FLIGHT BREAKDOWN (using the ORIGINAL seed=42 classifier from check_lowrul_classifier.py)")
    print("=" * 78)

    # Refit the original (non-bootstrapped) seed-42 classifier to match the prior result exactly
    F_train, y_train_bin, _ = build_matrix(X_train, y_train, models, scaler, dropped_idx_list)
    clf_base = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf_base.fit(F_train, y_train_bin)
    val_preds_base = clf_base.predict(F_val)

    print("\n-- Failing windows: catch rate by flight --")
    failing_idx = np.where(overshoot_mask)[0]
    by_flight_failing = defaultdict(lambda: {"total": 0, "caught": 0})
    for i in failing_idx:
        fid = val_flight_ids_arr[i]
        by_flight_failing[fid]["total"] += 1
        if val_preds_base[i] == 1:
            by_flight_failing[fid]["caught"] += 1
    for fid, d in sorted(by_flight_failing.items()):
        print(f"  flight {fid}: {d['caught']}/{d['total']} caught")

    print("\n-- Bucket(50,100) false positives: distribution by flight --")
    fp_idx = np.where(bucket_50_100_mask & (val_preds_base == 1))[0]
    by_flight_fp = defaultdict(int)
    by_flight_total_50_100 = defaultdict(int)
    for i in np.where(bucket_50_100_mask)[0]:
        by_flight_total_50_100[val_flight_ids_arr[i]] += 1
    for i in fp_idx:
        by_flight_fp[val_flight_ids_arr[i]] += 1
    for fid in sorted(by_flight_total_50_100):
        fp_count = by_flight_fp.get(fid, 0)
        total = by_flight_total_50_100[fid]
        print(f"  flight {fid}: {fp_count}/{total} flagged ({100*fp_count/total:.0f}%)")

    print("\nSTOP-OR-CONTINUE: if recall/FP swing widely across seeds, or either result is")
    print("concentrated on 1-2 flights rather than spread out, treat this the same as the")
    print("reweighting experiment -- looks fine in aggregate, isn't a general fix. Report back")
    print("before wiring anything into apply_calibration().")


if __name__ == "__main__":
    main()