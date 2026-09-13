"""
refit_calibration.py -- re-fits Step B calibration against updated
BUCKET_EDGES (merged (250,350) into (200,250) -> (200,350)) WITHOUT
retraining the ensemble itself. The trained variant models don't change;
only the bias-correction/conformal-width lookup tables built on top of
them do.

Rebuilds the IDENTICAL train/val/test split train_rul_ensemble.py's
__main__ used (same flights on disk, same random_state=42), so this is
a fair refit -- not fit on data the models have already seen as "train".

Run: python models/rul/refit_calibration.py   (from project root)
"""
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rul import load_all_flights, MODEL_OUT_DIR
from train_rul_ensemble import build_windowed_dataset_v5, load_ensemble
from calibrate_rul import fit_calibration, check_per_bucket_coverage, BUCKET_EDGES

if __name__ == "__main__":
    print(f"Refitting calibration with updated BUCKET_EDGES: {BUCKET_EDGES}\n")

    print("Loading flights and rebuilding the IDENTICAL split used during training ...")
    flights = load_all_flights()
    X, y, flight_ids = build_windowed_dataset_v5(flights)

    unique_flights = np.unique(flight_ids)
    train_flights, temp_flights = train_test_split(unique_flights, test_size=0.3, random_state=42)
    val_flights, test_flights = train_test_split(temp_flights, test_size=0.5, random_state=42)

    val_mask = np.isin(flight_ids, val_flights)
    test_mask = np.isin(flight_ids, test_flights)
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"Val windows: {len(X_val)} -- Test windows: {len(X_test)}\n")

    print("Loading trained ensemble (models unchanged, only calibration is refit) ...")
    models, scaler, dropped_idx_list = load_ensemble()

    print("\nRefitting calibration on VALIDATION set with merged buckets ...")
    params = fit_calibration(X_val, y_val, models, scaler, dropped_idx_list)

    print("\nRunning one-time coverage check on TEST set (untouched until now) ...")
    check_per_bucket_coverage(X_test, y_test, models, scaler, dropped_idx_list, params)