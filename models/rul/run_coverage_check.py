"""
run_coverage_check.py -- the ONE-TIME test-set acceptance check for Step B.

Run this once. Do not fold it into train_rul_ensemble.py's __main__ and do
not re-run it repeatedly to "tune" against test -- that defeats the point
of holding test out.

v9 fix: this used to run its OWN train_test_split(unique_flights, ...) on
raw flight_ids -- the old, leaky split. That's exactly the bug
train_rul_ensemble.py's v9 changelog warned about: two independent
reimplementations of split logic disagreeing about what's held out.
Now imports base_trajectory_id() directly from train_rul_ensemble.py and
reruns the SAME base-trajectory-grouped split (same random_state=42,
same two-step train_test_split), so this file's test_flights is
guaranteed identical to what training actually held out.

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
    RULRegressorVariant, build_windowed_dataset_v5, HIDDEN_SIZES, N_VARIANTS,
    base_trajectory_id,
)
from calibrate_rul import load_calibration_params, check_per_bucket_coverage

if __name__ == "__main__":
    flights = load_all_flights()
    X, y, flight_ids = build_windowed_dataset_v5(flights)

    # -----------------------------------------------------------------
    # v9: split on BASE TRAJECTORY IDs, identical logic to
    # train_rul_ensemble.py's __main__ block. Do not reimplement this
    # differently here -- import base_trajectory_id, don't redefine it.
    # -----------------------------------------------------------------
    unique_flights = np.unique(flight_ids)
    base_ids = np.array([base_trajectory_id(f) for f in unique_flights])
    unique_bases = np.unique(base_ids)
    print(f"Unique flight_ids: {len(unique_flights)}  ->  unique BASE trajectories: {len(unique_bases)}")

    train_bases, temp_bases = train_test_split(unique_bases, test_size=0.3, random_state=42)
    val_bases, test_bases = train_test_split(temp_bases, test_size=0.5, random_state=42)
    train_bases_set, val_bases_set, test_bases_set = set(train_bases), set(val_bases), set(test_bases)

    train_flights = np.array([f for f in unique_flights if base_trajectory_id(f) in train_bases_set])
    val_flights = np.array([f for f in unique_flights if base_trajectory_id(f) in val_bases_set])
    test_flights = np.array([f for f in unique_flights if base_trajectory_id(f) in test_bases_set])

    # Same hard sanity check as training -- fail loudly rather than
    # silently ship a second leaky split.
    split_of = {f: "train" for f in train_flights}
    split_of.update({f: "val" for f in val_flights})
    split_of.update({f: "test" for f in test_flights})
    crossing = 0
    for b in unique_bases:
        members = [f for f in unique_flights if base_trajectory_id(f) == b]
        if len(set(split_of[f] for f in members)) > 1:
            crossing += 1
    print(f"Base trajectories crossing splits: {crossing}  (should be 0)")
    assert crossing == 0, "Split still leaks across base trajectories -- do not proceed."

    test_mask = np.isin(flight_ids, test_flights)
    X_test, y_test = X[test_mask], y[test_mask]
    print(f"Test flights: {len(test_flights)}  Test bases: {len(test_bases)}")
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