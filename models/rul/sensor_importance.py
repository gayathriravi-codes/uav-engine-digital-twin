"""
sensor_importance.py -- Ranks sensors by importance for RUL prediction.

Built by Ashmitha, for the RUL ensemble tuning pass. This answers "which
sensors matter most for predicting REMAINING USEFUL LIFE specifically" --
NOT fault classification. Gayatri's fault classifier may find different
sensors important for identifying WHICH fault occurred; that's a different
question and this script does not answer it. The output here (RUL_CORE_SENSORS)
should only be used for RUL-related decisions (e.g. which sensors the RUL
ensemble should never drop two-of-together in feature bagging).

Method:
  1. Load flights + build the same windowed dataset the RUL ensemble uses
  2. Split by flight into train/val (same split as train_rul_ensemble.py,
     same random_state, so this is directly comparable)
  3. Flatten each window (30 timesteps x 7 sensors -> 210 features) since
     RandomForest doesn't take sequential input
  4. Train a RandomForestRegressor on flattened TRAIN windows, target =
     true_rul_timesteps
  5. Get feature_importances_ (210 values), then SUM the 30 timestep-columns
     belonging to each sensor to get one importance score per sensor
  6. Report ranked list -- this becomes RUL_CORE_SENSORS

Run: python models/rul/sensor_importance.py   (from project root, AeroTwin/)
"""
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rul import load_all_flights, build_windowed_dataset

from schema import SENSOR_FIELDS, WINDOW_SIZE


def flatten_windows(X):
    """
    X: shape (n_windows, window_size, n_sensors)
    Returns: shape (n_windows, window_size * n_sensors), flattened per window
    with timestep-major ordering (t0_sensor0, t0_sensor1, ..., t1_sensor0, ...)
    """
    n_windows = X.shape[0]
    return X.reshape(n_windows, -1)


def aggregate_importance_by_sensor(importances, window_size, sensor_fields):
    """
    importances: array of length window_size * n_sensors (flat, timestep-major)
    Returns: dict {sensor_name: summed_importance}
    """
    n_sensors = len(sensor_fields)
    importances = importances.reshape(window_size, n_sensors)
    per_sensor = importances.sum(axis=0)  # sum across all 30 timesteps
    return {sensor_fields[i]: float(per_sensor[i]) for i in range(n_sensors)}


if __name__ == "__main__":
    print("Loading flights from data/raw/ ...")
    flights = load_all_flights()
    print(f"Loaded {len(flights)} fault flights.")

    print("Building windowed dataset ...")
    X, y, flight_ids = build_windowed_dataset(flights)
    print(f"Total windows: {len(X)}")

    print("Splitting by flight into train/val (same split as ensemble script) ...")
    unique_flights = np.unique(flight_ids)
    train_flights, temp_flights = train_test_split(unique_flights, test_size=0.3, random_state=42)
    val_flights, test_flights = train_test_split(temp_flights, test_size=0.5, random_state=42)

    train_mask = np.isin(flight_ids, train_flights)
    val_mask = np.isin(flight_ids, val_flights)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    print(f"Train windows: {len(X_train)} -- Val windows: {len(X_val)}")

    print("Flattening windows for RandomForest (30 timesteps x 7 sensors -> 210 features) ...")
    X_train_flat = flatten_windows(X_train)
    X_val_flat = flatten_windows(X_val)

    print("Training RandomForestRegressor on flattened TRAIN windows (target: true_rul_timesteps) ...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train_flat, y_train)

    val_score = rf.score(X_val_flat, y_val)
    print(f"RandomForest R^2 on validation set: {val_score:.3f}  (sanity check, not the main output)")

    print("\nAggregating per-timestep importances up to per-sensor importance ...")
    per_sensor_importance = aggregate_importance_by_sensor(
        rf.feature_importances_, WINDOW_SIZE, SENSOR_FIELDS
    )

    ranked = sorted(per_sensor_importance.items(), key=lambda kv: kv[1], reverse=True)

    print("\n" + "=" * 50)
    print("RUL_CORE_SENSORS -- ranked by importance for RUL prediction")
    print("(NOT the same as fault-classification importance -- see docstring)")
    print("=" * 50)
    for rank, (sensor, importance) in enumerate(ranked, start=1):
        print(f"  {rank}. {sensor:<20s} {importance:.4f}")

    print("\nSuggested RUL_CORE_SENSORS constant (top half, rounded):")
    n_core = len(ranked) // 2 + 1  # top half, rounding up
    core_sensors = [s for s, _ in ranked[:n_core]]
    print(f"RUL_CORE_SENSORS = {core_sensors}")
    print("\nCopy this list into train_rul_ensemble.py's variant-grid redesign step --")
    print("feature bagging should never drop two of these together in the same variant.")