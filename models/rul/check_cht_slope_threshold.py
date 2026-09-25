"""
check_cht_slope_threshold.py

Quick, no-retrain check: does a cht_slope feature separate the 17 known
failing bucket-(0,50) validation windows from the rest of bucket (0,50)?
If yes, find a threshold and see what it would do to coverage if used as
an additional gate (same "is_ood" style override -- force lb=0 when
cht_slope < threshold), mirroring how the existing OOD z-distance gate
works, but checked HONESTLY (per-bucket breakdown, not just global pass).

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


def cht_slope(window):
    series = window[:, CHT_IDX]
    return (series[-1] - series[0]) / len(series)


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

    # Recompute coverage per window across ALL buckets (not just bucket 0),
    # plus cht_slope, so we can check this doesn't quietly break other buckets
    # the way the std-based OR-gate did.
    bucket_edges = [(0, 50), (50, 100), (100, 200), (200, 350)]

    records = []
    for i in range(len(X_val)):
        result = predict_rul_ensemble(X_val[i], models, scaler, dropped_idx_list, calibrated=True)
        true_rul = y_val[i]
        covered = result["rul_lower_bound_timesteps"] <= true_rul
        slope = cht_slope(X_val[i])
        records.append({
            "idx": i, "true_rul": true_rul, "covered": covered,
            "cht_slope": slope, "lb": result["rul_lower_bound_timesteps"],
        })

    # --- Step 1: does cht_slope separate failing vs passing WITHIN bucket (0,50)? ---
    bucket0 = [r for r in records if 0 <= r["true_rul"] < 50]
    failing = [r for r in bucket0 if not r["covered"]]
    passing = [r for r in bucket0 if r["covered"]]

    fail_slopes = [r["cht_slope"] for r in failing]
    pass_slopes = [r["cht_slope"] for r in passing]

    print(f"Bucket (0,50): {len(failing)} failing / {len(passing)} passing")
    print(f"  failing cht_slope: min={min(fail_slopes):.4f} max={max(fail_slopes):.4f} "
          f"mean={np.mean(fail_slopes):.4f}")
    print(f"  passing cht_slope: min={min(pass_slopes):.4f} max={max(pass_slopes):.4f} "
          f"mean={np.mean(pass_slopes):.4f}")

    # --- Step 2: sweep candidate thresholds, check global effect per bucket ---
    # Candidate: flag as "flat CHT" if cht_slope < threshold (failing windows
    # were near 0 / negative; healthy were strongly positive ~0.78).
    candidate_thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]

    print("\n=== Threshold sweep: flag if cht_slope < threshold, force lb=0 when flagged ===")
    for thresh in candidate_thresholds:
        print(f"\n--- threshold = {thresh} ---")
        total_flagged = 0
        for (lo, hi) in bucket_edges:
            bucket_records = [r for r in records if lo <= r["true_rul"] < hi]
            n = len(bucket_records)
            flagged = [r for r in bucket_records if r["cht_slope"] < thresh]
            total_flagged += len(flagged)

            # Recompute coverage AS IF flagged windows get lb=0 (always covers)
            n_covered = 0
            for r in bucket_records:
                if r in flagged:
                    n_covered += 1  # lb=0 always covers true_rul >= 0
                else:
                    n_covered += int(r["covered"])
            coverage = 100.0 * n_covered / n if n else float("nan")
            status = "PASS" if coverage >= 90.0 else "FAIL"
            print(f"  bucket {lo,hi}: coverage={coverage:5.1f}%  n={n:4d}  "
                  f"flagged={len(flagged):4d}  -> {status}")
        print(f"  TOTAL flagged: {total_flagged} / {len(records)}")


if __name__ == "__main__":
    main()