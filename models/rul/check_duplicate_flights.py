"""
check_duplicate_flights.py

Are specific flight_id pairs (e.g. cooling_degradation_009 / sensor_drift_009,
sensor_drift_016 / vibration_fault_016) actually near-duplicate raw trajectories
under different fault labels? Compares full per-window raw sensor values
(not just channel means) between suspected pairs, for the val split.

Val-only. Does not touch test.
Run: python check_duplicate_flights.py
"""
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rul import load_all_flights
from train_rul_ensemble import build_windowed_dataset_v5


def compare_pair(fid_a, fid_b, X_val, val_flight_ids_arr, y_val):
    a = X_val[val_flight_ids_arr == fid_a]
    b = X_val[val_flight_ids_arr == fid_b]
    y_a = y_val[val_flight_ids_arr == fid_a]
    y_b = y_val[val_flight_ids_arr == fid_b]
    print(f"\n{fid_a} vs {fid_b}: n={len(a)} vs {len(b)}")
    if len(a) == 0 or len(b) == 0:
        print("  one side missing from this val split -- skip")
        return
    n = min(len(a), len(b))
    for i in range(n):
        diff = np.abs(a[i] - b[i]).mean()
        print(f"  window {i}: mean abs diff = {diff:.5f}  "
              f"(y_a={y_a[i]:.1f}, y_b={y_b[i]:.1f})")


def main():
    print("Loading flights and building windowed dataset ...")
    flights = load_all_flights()
    X, y, flight_ids = build_windowed_dataset_v5(flights)

    unique_flights = np.unique(flight_ids)
    train_flights, temp_flights = train_test_split(unique_flights, test_size=0.3, random_state=42)
    val_flights, test_flights = train_test_split(temp_flights, test_size=0.5, random_state=42)
    val_mask = np.isin(flight_ids, val_flights)

    X_val, y_val = X[val_mask], y[val_mask]
    val_flight_ids_arr = flight_ids[val_mask]

    pairs = [
        ("cooling_degradation_009", "sensor_drift_009"),
        ("oil_issue_009", "sensor_drift_009"),
        ("sensor_drift_016", "vibration_fault_016"),
    ]
    for fid_a, fid_b in pairs:
        compare_pair(fid_a, fid_b, X_val, val_flight_ids_arr, y_val)

    print("\nAlso checking: are there OTHER flight_id pairs sharing an identical")
    print("or near-identical (cht,egt,rpm,vib) window-0 fingerprint, beyond the")
    print("three suspected pairs above? This catches duplication we haven't")
    print("manually spotted yet.")
    fingerprints = {}
    for fid in np.unique(val_flight_ids_arr):
        first_window = X_val[val_flight_ids_arr == fid][0]
        fp = tuple(np.round(first_window.mean(axis=0), 3))  # per-channel mean of window 0
        fingerprints.setdefault(fp, []).append(fid)
    for fp, fids in fingerprints.items():
        if len(fids) > 1:
            print(f"  MATCH: {fids}  fingerprint={fp}")


if __name__ == "__main__":
    main()