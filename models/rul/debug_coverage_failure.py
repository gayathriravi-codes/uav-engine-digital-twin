"""
debug_coverage_failure.py -- finds the exact test window and values that
trigger the "lb > y_cal" assertion in check_per_bucket_coverage, instead
of just seeing the traceback. Leading hypothesis: NaN in point_estimate
(NaN comparisons are always False, so `NaN <= NaN + 1e-6` fails even
though nothing was technically "exceeded").

Run: python models\rul\debug_coverage_failure.py   (from project root, AeroTwin/)
"""
import sys, os
sys.path.insert(0, 'models/rul')
import numpy as np
from sklearn.model_selection import train_test_split

from train_rul_ensemble import load_ensemble, load_all_flights, build_windowed_dataset_v5, predict_rul_ensemble
from calibrate_rul import load_calibration_params, apply_calibration, _bucket_index, BUCKET_EDGES

print("Loading ensemble ...")
models, scaler, dropped_idx_list = load_ensemble()
params = load_calibration_params()

print("Rebuilding test set ...")
flights = load_all_flights()
X, y, flight_ids = build_windowed_dataset_v5(flights)
unique_flights = np.unique(flight_ids)
train_flights, temp_flights = train_test_split(unique_flights, test_size=0.3, random_state=42)
val_flights, test_flights = train_test_split(temp_flights, test_size=0.5, random_state=42)
test_mask = np.isin(flight_ids, test_flights)
X_test, y_test = X[test_mask], y[test_mask]

print(f"Scanning {len(X_test)} test windows for the failing case ...")
n_nan = 0
n_bad = 0
for idx in range(len(X_test)):
    point_estimate = predict_rul_ensemble(
        X_test[idx], models, scaler, dropped_idx_list, calibrated=False
    )["point_estimate_timesteps"]
    y_cal, lb = apply_calibration(point_estimate, params)

    if np.isnan(point_estimate) or np.isnan(y_cal) or np.isnan(lb):
        n_nan += 1
        if n_nan <= 3:
            print(f"  [NaN] idx={idx}  true_rul={y_test[idx]:.1f}  "
                  f"point_estimate={point_estimate}  y_cal={y_cal}  lb={lb}")
        continue

    if lb > y_cal + 1e-6:
        n_bad += 1
        if n_bad <= 3:
            b0 = _bucket_index(point_estimate, BUCKET_EDGES)
            print(f"  [BAD] idx={idx}  true_rul={y_test[idx]:.1f}  "
                  f"point_estimate={point_estimate:.4f}  y_cal={y_cal:.4f}  lb={lb:.4f}  "
                  f"(lb - y_cal = {lb - y_cal:.6f}, initial bucket={BUCKET_EDGES[b0]})")

print(f"\nTotal windows scanned: {len(X_test)}")
print(f"NaN cases: {n_nan}")
print(f"Non-NaN lb>y_cal cases: {n_bad}")