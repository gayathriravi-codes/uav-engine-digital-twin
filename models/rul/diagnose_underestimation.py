"""
diagnose_underestimation.py -- Checks RUL underestimation bias AND whether
the ensemble's own uncertainty (std across variants) correctly widens in
the region where errors are largest.

v2: added per-bucket std tracking. The question this answers: is the
ensemble's confidence band doing its job (widening where predictions are
less trustworthy), or is it flat/uninformative regardless of how wrong
the point estimate is?

v3 (this version): updated to use build_windowed_dataset_v5 instead of the
old 7-column build_windowed_dataset, since the ensemble models now expect
the 10-column featurized window (7 raw sensors + 3 derived features from
Step A). Windows from build_windowed_dataset_v5 already include the
derived features, so no extra build_inference_window() call is needed
inside the bucket loop below.

Run: python models/rul/diagnose_underestimation.py   (from project root)
"""
import os
import sys

import numpy as np
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rul import load_all_flights, MODEL_OUT_DIR
from train_rul_ensemble import (
    RULRegressorVariant, predict_rul_ensemble, build_windowed_dataset_v5,
    build_inference_window, HIDDEN_SIZES, N_VARIANTS
)
import joblib

from schema import SENSOR_FIELDS


def describe(name, arr):
    print(f"  {name:<12s} n={len(arr):5d}  mean={arr.mean():7.1f}  median={np.median(arr):7.1f}  "
          f"min={arr.min():6.1f}  max={arr.max():6.1f}  std={arr.std():6.1f}")


if __name__ == "__main__":
    print("Loading flights and building windows (same pipeline as ensemble) ...")
    flights = load_all_flights()
    X, y, flight_ids = build_windowed_dataset_v5(flights)

    unique_flights = np.unique(flight_ids)
    train_flights, temp_flights = train_test_split(unique_flights, test_size=0.3, random_state=42)
    val_flights, test_flights = train_test_split(temp_flights, test_size=0.5, random_state=42)

    train_mask = np.isin(flight_ids, train_flights)
    val_mask = np.isin(flight_ids, val_flights)
    test_mask = np.isin(flight_ids, test_flights)

    y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]
    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]

    print("\n" + "=" * 70)
    print("QUESTION 1 & 2: RUL label distribution across splits")
    print("=" * 70)
    describe("train", y_train)
    describe("val", y_val)
    describe("test", y_test)

    print("\nLoading trained ensemble ...")
    scaler = joblib.load(os.path.join(MODEL_OUT_DIR, "rul_ensemble_scaler.joblib"))
    dropped_idx_list = joblib.load(os.path.join(MODEL_OUT_DIR, "rul_ensemble_dropped_idx.joblib"))

    models = []
    for i in range(N_VARIANTS):
        model = RULRegressorVariant(hidden_size=HIDDEN_SIZES[i])
        state_path = os.path.join(MODEL_OUT_DIR, f"rul_model_variant_{i}.pt")
        model.load_state_dict(torch.load(state_path))
        model.eval()
        models.append(model)

    buckets = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, 350)]

    print("\n" + "=" * 70)
    print("QUESTION 3: prediction error AND ensemble std, by RUL range (VALIDATION set)")
    print("=" * 70)
    print(f"\n{'RUL range':<15s}{'n':>6s}{'mean_true':>12s}{'mean_pred':>12s}"
          f"{'mean_error':>13s}{'mean_std':>11s}")

    bucket_results = []
    for lo, hi in buckets:
        bucket_mask = (y_val >= lo) & (y_val < hi)
        if bucket_mask.sum() == 0:
            continue
        idxs = np.where(bucket_mask)[0]
        errors = []
        preds = []
        stds = []
        for idx in idxs:
            result = predict_rul_ensemble(X_val[idx], models, scaler, dropped_idx_list)
            preds.append(result["point_estimate_timesteps"])
            errors.append(result["point_estimate_timesteps"] - y_val[idx])
            stds.append(result["std_timesteps"])
        errors = np.array(errors)
        preds = np.array(preds)
        stds = np.array(stds)
        print(f"{lo}-{hi:<10d}{bucket_mask.sum():>6d}{y_val[idxs].mean():>12.1f}"
              f"{preds.mean():>12.1f}{errors.mean():>13.1f}{stds.mean():>11.1f}")
        bucket_results.append((lo, hi, errors.mean(), stds.mean()))

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    # Compare std of the two most-biased high buckets against the two least-biased low buckets
    low_stds = [s for (lo, hi, e, s) in bucket_results if hi <= 100]
    high_stds = [s for (lo, hi, e, s) in bucket_results if lo >= 200]
    if low_stds and high_stds:
        low_mean = np.mean(low_stds)
        high_mean = np.mean(high_stds)
        print(f"Mean std, low-RUL buckets (<=100):  {low_mean:.1f}")
        print(f"Mean std, high-RUL buckets (>=200): {high_mean:.1f}")
        if high_mean > low_mean * 1.3:
            print("\n-> WIDENS WITH RUL: the ensemble's own uncertainty correctly grows where")
            print("   error is largest. This confirms option (c) -- present the wider band")
            print("   as the confidence-bound feature working as designed. No correction needed.")
        else:
            print("\n-> FLAT: the ensemble's std does NOT meaningfully widen despite much larger")
            print("   errors in the high-RUL region. The band is underconfident about its own")
            print("   underconfidence. Needs either more diversity targeted at high-RUL windows,")
            print("   or fall back to documenting this as a plain limitation (option b).")
    else:
        print("Not enough buckets on one side to compare -- inspect the table above manually.")