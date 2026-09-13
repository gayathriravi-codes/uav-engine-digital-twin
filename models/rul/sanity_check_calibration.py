"""
sanity_check_calibration.py -- eyeball check on a handful of VALIDATION
windows, comparing raw vs calibrated predict_rul_ensemble output.
Not the acceptance test (that's check_per_bucket_coverage on TEST) --
this is just a quick "did the wiring work" look before spending the
one-time test check.

Run: python models\\rul\\sanity_check_calibration.py
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
    RULRegressorVariant, predict_rul_ensemble, build_windowed_dataset_v5,
    HIDDEN_SIZES, N_VARIANTS
)

if __name__ == "__main__":
    flights = load_all_flights()
    X, y, flight_ids = build_windowed_dataset_v5(flights)

    unique_flights = np.unique(flight_ids)
    train_flights, temp_flights = train_test_split(unique_flights, test_size=0.3, random_state=42)
    val_flights, test_flights = train_test_split(temp_flights, test_size=0.5, random_state=42)
    val_mask = np.isin(flight_ids, val_flights)
    X_val, y_val = X[val_mask], y[val_mask]

    scaler = joblib.load(os.path.join(MODEL_OUT_DIR, "rul_ensemble_scaler.joblib"))
    dropped_idx_list = joblib.load(os.path.join(MODEL_OUT_DIR, "rul_ensemble_dropped_idx.joblib"))
    models = []
    for i in range(N_VARIANTS):
        model = RULRegressorVariant(hidden_size=HIDDEN_SIZES[i])
        model.load_state_dict(torch.load(os.path.join(MODEL_OUT_DIR, f"rul_model_variant_{i}.pt")))
        model.eval()
        models.append(model)

    # Pick a spread of true-RUL values across buckets, not just the first 5.
    sample_idxs = np.linspace(0, len(X_val) - 1, 10).astype(int)

    print(f"{'true':>7s} {'raw_pt':>8s} {'raw_lb':>8s} | {'cal_pt':>8s} {'cal_lb':>8s}")
    for idx in sample_idxs:
        raw = predict_rul_ensemble(X_val[idx], models, scaler, dropped_idx_list, calibrated=False)
        cal = predict_rul_ensemble(X_val[idx], models, scaler, dropped_idx_list, calibrated=True)
        print(f"{y_val[idx]:7.1f} {raw['point_estimate_minutes']:8.1f} {raw['rul_lower_bound_minutes']:8.1f} | "
              f"{cal['point_estimate_minutes']:8.1f} {cal['rul_lower_bound_minutes']:8.1f}")