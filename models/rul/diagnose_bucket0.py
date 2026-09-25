"""
diagnose_bucket0.py -- ONE-TIME diagnostic run alongside run_coverage_check.py.

Same test split, same saved models, same calibration params as
run_coverage_check.py -- but for every window whose TRUE RUL falls in
bucket 0 (0,50), it also logs which bucket's calibration band
apply_calibration() actually pulled from (b1), so we can tell apart:

  (a) band came from bucket 0 and was still too narrow -> override the
      quantile further, or widen bucket 0's band directly
  (b) band came from a DIFFERENT bucket (mismatch between true-value
      bucketing used for evaluation and estimate-based bucketing used
      for lookup) -> the 0.94 override on bucket 0 never touched these
      windows, which is why it barely moved coverage

Run once: python diagnose_bucket0.py
"""
import os
import sys

import numpy as np
import torch
from sklearn.model_selection import train_test_split
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rul import load_all_flights, MODEL_OUT_DIR
from train_rul_ensemble import (
    RULRegressorVariant, build_windowed_dataset_v5, HIDDEN_SIZES, N_VARIANTS,
    predict_rul_ensemble,
)
from calibrate_rul import load_calibration_params, _bucket_index, BUCKET_EDGES

if __name__ == "__main__":
    flights = load_all_flights()
    X, y, flight_ids = build_windowed_dataset_v5(flights)

    unique_flights = np.unique(flight_ids)
    train_flights, temp_flights = train_test_split(unique_flights, test_size=0.3, random_state=42)
    val_flights, test_flights = train_test_split(temp_flights, test_size=0.5, random_state=42)
    test_mask = np.isin(flight_ids, test_flights)
    X_test, y_test = X[test_mask], y[test_mask]
    print(f"Test windows: {len(X_test)} (same split as run_coverage_check.py)")

    scaler = joblib.load(os.path.join(MODEL_OUT_DIR, "rul_ensemble_scaler.joblib"))
    dropped_idx_list = joblib.load(os.path.join(MODEL_OUT_DIR, "rul_ensemble_dropped_idx.joblib"))
    models = []
    for i in range(N_VARIANTS):
        model = RULRegressorVariant(hidden_size=HIDDEN_SIZES[i])
        model.load_state_dict(torch.load(os.path.join(MODEL_OUT_DIR, f"rul_model_variant_{i}.pt")))
        model.eval()
        models.append(model)

    params = load_calibration_params()
    rough_buckets = params["rough_buckets"]
    buckets = params["buckets"]

    y_test = np.asarray(y_test, dtype=float)
    true_bucket_ids = np.array([_bucket_index(v, BUCKET_EDGES) for v in y_test])

    b = 0  # bucket (0, 50)
    idxs = np.where(true_bucket_ids == b)[0]
    print(f"\n{len(idxs)} test windows with TRUE RUL in bucket {BUCKET_EDGES[b]}\n")

    mismatch = 0
    own_bucket_fail = 0
    own_bucket_pass = 0
    mismatch_fail = 0
    mismatch_pass = 0

    for idx in idxs:
        point_estimate = predict_rul_ensemble(
            X_test[idx], models, scaler, dropped_idx_list, calibrated=False
        )["point_estimate_timesteps"]

        # Replicate apply_calibration's two-pass lookup, but log b1.
        b0 = _bucket_index(point_estimate, BUCKET_EDGES)
        bias0 = rough_buckets.get(b0, {"mean_residual": 0.0})["mean_residual"]
        rough_corrected = point_estimate + bias0
        b1 = _bucket_index(rough_corrected, BUCKET_EDGES)

        info = buckets.get(b1, {"mean_residual": 0.0, "conformal_width": 0.0})
        y_cal = point_estimate + info["mean_residual"]
        y_cal = max(y_cal, 0.0)
        lb = y_cal - info["conformal_width"]
        lb = min(lb, y_cal)
        lb = max(lb, 0.0)

        covered = y_test[idx] >= lb

        if b1 != b:
            mismatch += 1
            if covered:
                mismatch_pass += 1
            else:
                mismatch_fail += 1
            print(f"  MISMATCH  true_rul={y_test[idx]:6.1f}  point_est={point_estimate:6.1f}  "
                  f"band_from_bucket={BUCKET_EDGES[b1]}  lb={lb:6.1f}  covered={covered}")
        else:
            if covered:
                own_bucket_pass += 1
            else:
                own_bucket_fail += 1

    print(f"\n--- Summary for true bucket {BUCKET_EDGES[b]} ---")
    print(f"  Total windows: {len(idxs)}")
    print(f"  Got bucket 0's own band:      {own_bucket_pass + own_bucket_fail:4d}  "
          f"(covered={own_bucket_pass}, missed={own_bucket_fail})")
    print(f"  Got a DIFFERENT bucket's band: {mismatch:4d}  "
          f"(covered={mismatch_pass}, missed={mismatch_fail})")
    print(f"\n  Coverage using own band only:  "
          f"{own_bucket_pass/(own_bucket_pass+own_bucket_fail)*100 if (own_bucket_pass+own_bucket_fail) else float('nan'):.1f}%")
    if mismatch:
        print(f"  Coverage on mismatched windows: {mismatch_pass/mismatch*100:.1f}%")