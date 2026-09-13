"""
run_coverage_check.py -- the ONE-TIME test-set acceptance check for Step B.

Run this once. Do not fold it into train_rul_ensemble.py's __main__ and do
not re-run it repeatedly to "tune" against test -- that defeats the point
of holding test out.

Run: python models\\rul\\run_coverage_check.py
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
    RULRegressorVariant, build_windowed_dataset_v5, HIDDEN_SIZES, N_VARIANTS
)
from calibrate_rul import load_calibration_params, check_per_bucket_coverage

if __name__ == "__main__":
    flights = load_all_flights()
    X, y, flight_ids = build_windowed_dataset_v5(flights)

    unique_flights = np.unique(flight_ids)
    train_flights, temp_flights = train_test_split(unique_flights, test_size=0.3, random_state=42)
    val_flights, test_flights = train_test_split(temp_flights, test_size=0.5, random_state=42)
    test_mask = np.isin(flight_ids, test_flights)
    X_test, y_test = X[test_mask], y[test_mask]
    print(f"Test windows: {len(X_test)} (touched for the first time, right now)")

    scaler = joblib.load(os.path.join(MODEL_OUT_DIR, "rul_ensemble_scaler.joblib"))
    dropped_idx_list = joblib.load(os.path.join(MODEL_OUT_DIR, "rul_ensemble_dropped_idx.joblib"))
    models = []
    for i in range(N_VARIANTS):
        model = RULRegressorVariant(hidden_size=HIDDEN_SIZES[i])
        model.load_state_dict(torch.load(os.path.join(MODEL_OUT_DIR, f"rul_model_variant_{i}.pt")))
        model.eval()
        models.append(model)

    params = load_calibration_params()
    check_per_bucket_coverage(X_test, y_test, models, scaler, dropped_idx_list, params)