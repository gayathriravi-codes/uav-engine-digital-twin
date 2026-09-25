"""
check_coverage_now.py -- independent re-run of check_per_bucket_coverage
against the CURRENTLY SAVED ensemble artifacts, without retraining.

Rebuilds X_test/y_test the exact same way train_rul_ensemble.py's __main__
block does (same flight split, same random_state=42), then loads the
already-trained-and-saved ensemble via load_ensemble() instead of
retraining, then calls check_per_bucket_coverage() once.

FIX (this version): check_per_bucket_coverage() internally calls
predict_rul_ensemble(), which already scales its input using the passed-in
scaler. The previous version of this script pre-scaled X_test itself before
also passing it into check_per_bucket_coverage(), causing every window to
be scaled TWICE. That silently compressed the tails of the distribution,
which inflated apparent coverage most on the extreme/near-failure windows
-- exactly bucket (0,50), exactly where the false "97.1% PASS" showed up
instead of the real 86.5% FAIL. Fix: pass raw (unscaled) X_test straight
through, same as run_coverage_check.py does.

Run from models/rul/:
    python check_coverage_now.py
"""
import numpy as np
from sklearn.model_selection import train_test_split

from train_rul_ensemble import (
    load_all_flights,
    build_windowed_dataset_v5,
    load_ensemble,
)
from calibrate_rul import load_calibration_params, check_per_bucket_coverage

print("Loading flights from data/raw/ ...")
flights = load_all_flights()
print(f"Loaded {len(flights)} fault flights.")

print("Building windowed dataset (same as train_rul_ensemble.py) ...")
X, y, flight_ids = build_windowed_dataset_v5(flights)

print("Splitting by flight (same random_state=42 as train_rul_ensemble.py) ...")
unique_flights = np.unique(flight_ids)
train_flights, temp_flights = train_test_split(unique_flights, test_size=0.3, random_state=42)
val_flights, test_flights = train_test_split(temp_flights, test_size=0.5, random_state=42)

test_mask = np.isin(flight_ids, test_flights)
X_test, y_test = X[test_mask], y[test_mask]
print(f"Test windows: {len(X_test)}  Test flights: {len(test_flights)}")

print("\nLoading the CURRENTLY SAVED ensemble (no retraining) ...")
models, scaler, dropped_idx_list = load_ensemble()

print("\nRunning check_per_bucket_coverage() against saved calibration params ...")
print("(NOTE: X_test passed RAW/unscaled -- check_per_bucket_coverage scales")
print(" internally via predict_rul_ensemble(). Do not pre-scale here.)")
params = load_calibration_params()
check_per_bucket_coverage(X_test, y_test, models, scaler, dropped_idx_list, params)