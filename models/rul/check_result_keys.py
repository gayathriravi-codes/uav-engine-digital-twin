"""
check_result_keys.py

One-off diagnostic: print the actual keys/values returned by
predict_rul_ensemble(..., calibrated=False) for a single window, so we
stop guessing key names (ood_zscore_distance was wrong last time).

Val-only, single window, does not touch test.
"""
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rul import load_all_flights
from train_rul_ensemble import build_windowed_dataset_v5, load_ensemble, predict_rul_ensemble


def main():
    flights = load_all_flights()
    X, y, flight_ids = build_windowed_dataset_v5(flights)

    unique_flights = np.unique(flight_ids)
    train_flights, temp_flights = train_test_split(unique_flights, test_size=0.3, random_state=42)
    val_flights, test_flights = train_test_split(temp_flights, test_size=0.5, random_state=42)
    val_mask = np.isin(flight_ids, val_flights)
    X_val, y_val = X[val_mask], y[val_mask]

    models, scaler, dropped_idx_list = load_ensemble()

    result = predict_rul_ensemble(X_val[0], models, scaler, dropped_idx_list, calibrated=False)
    print("Keys and values in predict_rul_ensemble(..., calibrated=False) result:\n")
    for k, v in result.items():
        print(f"  {k!r}: {v!r}  (type={type(v).__name__})")


if __name__ == "__main__":
    main()