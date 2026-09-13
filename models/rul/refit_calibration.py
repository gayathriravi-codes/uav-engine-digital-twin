"""
refit_calibration.py -- re-runs Step B calibration using the ALREADY-TRAINED
ensemble saved on disk, without retraining anything.

Use this after fixing the X_val_scaled -> X_val bug, instead of re-running
train_rul_ensemble.py (which would retrain all 7 variants from scratch).

Run: python models\\rul\\refit_calibration.py   (from project root, AeroTwin/)
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
from calibrate_rul import fit_calibration


if __name__ == "__main__":
    print("Rebuilding windowed dataset (same pipeline/seed as training, so the "
          "train/val/test split matches exactly) ...")
    flights = load_all_flights()
    X, y, flight_ids = build_windowed_dataset_v5(flights)

    unique_flights = np.unique(flight_ids)
    train_flights, temp_flights = train_test_split(unique_flights, test_size=0.3, random_state=42)
    val_flights, test_flights = train_test_split(temp_flights, test_size=0.5, random_state=42)

    val_mask = np.isin(flight_ids, val_flights)
    X_val, y_val = X[val_mask], y[val_mask]
    print(f"Val windows: {len(X_val)}")

    print("Loading saved ensemble artifacts (no retraining) ...")
    scaler = joblib.load(os.path.join(MODEL_OUT_DIR, "rul_ensemble_scaler.joblib"))
    dropped_idx_list = joblib.load(os.path.join(MODEL_OUT_DIR, "rul_ensemble_dropped_idx.joblib"))

    models = []
    for i in range(N_VARIANTS):
        model = RULRegressorVariant(hidden_size=HIDDEN_SIZES[i])
        state_path = os.path.join(MODEL_OUT_DIR, f"rul_model_variant_{i}.pt")
        model.load_state_dict(torch.load(state_path))
        model.eval()
        models.append(model)

    print("\nRe-fitting calibration on VALIDATION set (X_val is UNSCALED here -- "
          "predict_rul_ensemble scales internally, this is the fix) ...")
    fit_calibration(X_val, y_val, models, scaler, dropped_idx_list)