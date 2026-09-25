"""
check_2d_slope_separation.py

Quick, no-retrain check: does the COMBINATION of cht_slope and egt_slope
(or other simple pairwise combos) separate the 17 known-failing bucket-(0,50)
validation windows from the rest, even though neither slope alone does?

Val-only. Does not touch test. Safe to re-run.
"""
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rul import load_all_flights
from train_rul_ensemble import build_windowed_dataset_v5, load_ensemble, predict_rul_ensemble
from calibrate_rul import load_calibration_params
from schema import SENSOR_FIELDS

CHT_IDX = SENSOR_FIELDS.index("cht")
EGT_IDX = SENSOR_FIELDS.index("egt")
RPM_IDX = SENSOR_FIELDS.index("rpm")
VIB_IDX = SENSOR_FIELDS.index("vibration")


def slope(window, idx, n=10):
    series = window[:, idx]
    tail = series[-n:]
    return (tail[-1] - tail[0]) / len(tail)


def main():
    flights = load_all_flights()
    X, y, flight_ids = build_windowed_dataset_v5(flights)

    unique_flights = np.unique(flight_ids)
    train_flights, temp_flights = train_test_split(unique_flights, test_size=0.3, random_state=42)
    val_flights, test_flights = train_test_split(temp_flights, test_size=0.5, random_state=42)
    val_mask = np.isin(flight_ids, val_flights)
    X_val, y_val = X[val_mask], y[val_mask]

    models, scaler, dropped_idx_list = load_ensemble()
    params = load_calibration_params()

    bucket0_records = []
    for i in range(len(X_val)):
        true_rul = y_val[i]
        if not (0 <= true_rul < 50):
            continue
        result = predict_rul_ensemble(X_val[i], models, scaler, dropped_idx_list, calibrated=True)
        covered = result["rul_lower_bound_timesteps"] <= true_rul
        bucket0_records.append({
            "idx": i,
            "covered": covered,
            "cht_slope": slope(X_val[i], CHT_IDX),
            "egt_slope": slope(X_val[i], EGT_IDX),
            "rpm_slope": slope(X_val[i], RPM_IDX),
            "vib_slope": slope(X_val[i], VIB_IDX),
        })

    failing = [r for r in bucket0_records if not r["covered"]]
    passing = [r for r in bucket0_records if r["covered"]]
    print(f"Bucket (0,50): {len(failing)} failing / {len(passing)} passing\n")

    # Print full per-window feature vectors for failing windows -- eyeball for a pattern
    print("=== Failing windows: cht_slope, egt_slope, rpm_slope, vib_slope ===")
    for r in failing:
        print(f"  idx={r['idx']:4d}  cht={r['cht_slope']:7.4f}  egt={r['egt_slope']:7.4f}  "
              f"rpm={r['rpm_slope']:7.4f}  vib={r['vib_slope']:7.4f}")

    print("\n=== Passing windows (first 15 for comparison) ===")
    for r in passing[:15]:
        print(f"  idx={r['idx']:4d}  cht={r['cht_slope']:7.4f}  egt={r['egt_slope']:7.4f}  "
              f"rpm={r['rpm_slope']:7.4f}  vib={r['vib_slope']:7.4f}")

    # Try a few simple combined scores and see if any cleanly separates
    print("\n=== Combined score checks ===")
    combos = {
        "cht_slope + egt_slope": lambda r: r["cht_slope"] + r["egt_slope"],
        "cht_slope - rpm_slope": lambda r: r["cht_slope"] - r["rpm_slope"],
        "|cht_slope| + |egt_slope|": lambda r: abs(r["cht_slope"]) + abs(r["egt_slope"]),
        "cht_slope * rpm_slope": lambda r: r["cht_slope"] * r["rpm_slope"],
    }
    for name, fn in combos.items():
        fail_vals = [fn(r) for r in failing]
        pass_vals = [fn(r) for r in passing]
        overlap = max(0.0, min(max(fail_vals), max(pass_vals)) - max(min(fail_vals), min(pass_vals)))
        print(f"\n{name}:")
        print(f"  failing: min={min(fail_vals):.4f} max={max(fail_vals):.4f} mean={np.mean(fail_vals):.4f}")
        print(f"  passing: min={min(pass_vals):.4f} max={max(pass_vals):.4f} mean={np.mean(pass_vals):.4f}")
        print(f"  ranges {'OVERLAP' if (max(fail_vals) > min(pass_vals) and max(pass_vals) > min(fail_vals)) else 'may separate'}")


if __name__ == "__main__":
    main()