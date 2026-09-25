"""
diagnose_bucket_100_200_test.py

DIAGNOSTIC-ONLY look at test's bucket (100,200) failure (79.2% coverage,
from run_coverage_check.py's already-completed one-time acceptance run).

This is explicitly NOT a re-run of the acceptance check and NOT a tuning
pass -- calibration params and models are unchanged, nothing here feeds
back into any fit/refit step. It exists purely to read per-window detail
out of a result that was already obtained, because validation's bucket
(100,200) failure pattern (98.1% coverage, isolated to one base trajectory)
did not reproduce or explain test's much broader 79.2% failure.

Framed explicitly per the discipline this project has kept throughout:
touching test again is a real, deliberate exception here, done once, to
understand an existing result -- not to search for a fix that then gets
"confirmed" by re-running this. Any actual fix still gets validated on
val only, same as every other bucket investigation in this project.

Run: python models\\rul\\diagnose_bucket_100_200_test.py
"""
import os
import sys
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rul import load_all_flights
from train_rul_ensemble import (
    build_windowed_dataset_v5, load_ensemble, predict_rul_ensemble,
    base_trajectory_id,
)
from calibrate_rul import load_calibration_params

BUCKET_LO, BUCKET_HI = 100, 200


def main():
    print("Loading flights and building windowed dataset ...")
    flights = load_all_flights()
    X, y, flight_ids = build_windowed_dataset_v5(flights)

    # Identical split logic to train_rul_ensemble.py / run_coverage_check.py.
    unique_flights = np.unique(flight_ids)
    base_ids = np.array([base_trajectory_id(f) for f in unique_flights])
    unique_bases = np.unique(base_ids)

    train_bases, temp_bases = train_test_split(unique_bases, test_size=0.3, random_state=42)
    val_bases, test_bases = train_test_split(temp_bases, test_size=0.5, random_state=42)
    test_bases_set = set(test_bases)

    print(f"Test bases: {sorted(test_bases_set)}")

    test_flights = np.array([f for f in unique_flights if base_trajectory_id(f) in test_bases_set])
    test_mask = np.isin(flight_ids, test_flights)
    X_test, y_test = X[test_mask], y[test_mask]
    test_flight_ids_arr = flight_ids[test_mask]

    print(f"Test windows: {len(X_test)}  Test flights: {len(test_flights)}\n")

    models, scaler, dropped_idx_list = load_ensemble()
    load_calibration_params()

    print(f"Running calibrated ensemble on bucket ({BUCKET_LO},{BUCKET_HI}) TEST windows "
          f"(diagnostic read only, not a re-check) ...\n")

    records = []
    for i in range(len(X_test)):
        true_rul = y_test[i]
        if not (BUCKET_LO <= true_rul < BUCKET_HI):
            continue
        result = predict_rul_ensemble(X_test[i], models, scaler, dropped_idx_list, calibrated=True)
        point_est = result["point_estimate_timesteps"]
        lb = result["rul_lower_bound_timesteps"]
        std = result["std_timesteps"]
        covered = lb <= true_rul
        records.append({
            "idx": i,
            "flight": test_flight_ids_arr[i],
            "true_rul": true_rul,
            "point_est": point_est,
            "lb": lb,
            "std": std,
            "covered": covered,
            "gap": true_rul - lb,
        })

    failing = [r for r in records if not r["covered"]]
    passing = [r for r in records if r["covered"]]
    print(f"Bucket ({BUCKET_LO},{BUCKET_HI}): {len(failing)} failing / {len(passing)} passing "
          f"(coverage={100*len(passing)/len(records):.1f}%, n={len(records)})\n")

    print("=== Failing windows: true_rul, point_est, lower_bound, std, gap (true - lb) ===")
    print(f"{'idx':>5} {'flight':30s} {'true':>7} {'point_est':>10} {'lb':>7} {'std':>7} {'gap':>7}")
    for r in sorted(failing, key=lambda r: r["gap"]):
        print(f"{r['idx']:5d} {r['flight']:30s} {r['true_rul']:7.1f} {r['point_est']:10.1f} "
              f"{r['lb']:7.1f} {r['std']:7.1f} {r['gap']:7.1f}")

    print("\n=== Diagnosis: calibration (band too narrow) vs model accuracy ===")
    point_est_errors = [r["point_est"] - r["true_rul"] for r in failing]
    lb_overshoots = [r["lb"] - r["true_rul"] for r in failing]
    print(f"  Point-estimate error (point_est - true_rul): mean={np.mean(point_est_errors):+.2f}  "
          f"min={min(point_est_errors):+.2f}  max={max(point_est_errors):+.2f}")
    print(f"  Lower-bound overshoot (lb - true_rul): mean={np.mean(lb_overshoots):+.2f}  "
          f"min={min(lb_overshoots):+.2f}  max={max(lb_overshoots):+.2f}")
    point_est_also_over = sum(1 for e in point_est_errors if e > 0)
    print(f"  Point estimate ALSO overshoots true_rul in {point_est_also_over}/{len(failing)} failing windows.")

    print("\n=== Failing-window count by flight ===")
    by_flight = defaultdict(lambda: {"total": 0, "failing": 0})
    for r in records:
        by_flight[r["flight"]]["total"] += 1
        if not r["covered"]:
            by_flight[r["flight"]]["failing"] += 1
    for flight, d in sorted(by_flight.items()):
        if d["failing"] > 0:
            print(f"  {flight:30s}: {d['failing']}/{d['total']} failing")

    print("\n=== Same breakdown, grouped by base trajectory ===")
    by_base = defaultdict(lambda: {"total": 0, "failing": 0})
    for r in records:
        base = base_trajectory_id(r["flight"])
        by_base[base]["total"] += 1
        if not r["covered"]:
            by_base[base]["failing"] += 1
    for base, d in sorted(by_base.items()):
        if d["failing"] > 0:
            print(f"  base {base}: {d['failing']}/{d['total']} failing")

    print("\nThis is a read of an existing result -- no fit/refit happened here. Whatever")
    print("this shows, the next step is to reproduce/address it on VALIDATION data before")
    print("touching test again for any actual re-check.")


if __name__ == "__main__":
    main()