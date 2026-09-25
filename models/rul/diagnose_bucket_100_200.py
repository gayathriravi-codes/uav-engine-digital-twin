"""
diagnose_bucket_100_200.py

Per-window diagnostic for the bucket (100,200) test-coverage failure
(79.2%, first surfaced on the corrected leakage-free test run). Same
style as the original diagnose_bucket0.py: run the trained, leakage-free
ensemble on VALIDATION (not test -- test is spent for this cycle), isolate
failing windows in this bucket, and print enough per-window detail to tell
apart a calibration problem (band too narrow) from a genuine model-accuracy
problem (point estimate itself wrong), same distinction diagnose_bucket0.py
made for the original bucket.

Also breaks down failures by flight/base-trajectory, since the reweighting
experiment earlier in this investigation showed aggregate numbers can hide
a collapse concentrated on 1-2 flights.

Val-only. Does NOT touch test.
Run: python models\\rul\\diagnose_bucket_100_200.py
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

    # Same base-trajectory-grouped split as train_rul_ensemble.py / the
    # corrected run_coverage_check.py -- imported, not reimplemented.
    unique_flights = np.unique(flight_ids)
    base_ids = np.array([base_trajectory_id(f) for f in unique_flights])
    unique_bases = np.unique(base_ids)

    train_bases, temp_bases = train_test_split(unique_bases, test_size=0.3, random_state=42)
    val_bases, test_bases = train_test_split(temp_bases, test_size=0.5, random_state=42)
    val_bases_set = set(val_bases)

    val_flights = np.array([f for f in unique_flights if base_trajectory_id(f) in val_bases_set])
    val_mask = np.isin(flight_ids, val_flights)
    X_val, y_val = X[val_mask], y[val_mask]
    val_flight_ids_arr = flight_ids[val_mask]

    print(f"Val windows: {len(X_val)}  Val flights: {len(val_flights)}\n")

    models, scaler, dropped_idx_list = load_ensemble()
    load_calibration_params()  # not used directly; keeps parity with pipeline

    print(f"Running calibrated ensemble on bucket ({BUCKET_LO},{BUCKET_HI}) val windows ...\n")

    records = []
    for i in range(len(X_val)):
        true_rul = y_val[i]
        if not (BUCKET_LO <= true_rul < BUCKET_HI):
            continue
        result = predict_rul_ensemble(X_val[i], models, scaler, dropped_idx_list, calibrated=True)
        point_est = result["point_estimate_timesteps"]
        lb = result["rul_lower_bound_timesteps"]
        std = result["std_timesteps"]
        covered = lb <= true_rul
        records.append({
            "idx": i,
            "flight": val_flight_ids_arr[i],
            "true_rul": true_rul,
            "point_est": point_est,
            "lb": lb,
            "std": std,
            "covered": covered,
            "gap": true_rul - lb,  # positive = lb correctly below true; negative = lb overshot true
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

    print("\n=== Passing windows (first 15, for comparison) ===")
    print(f"{'idx':>5} {'flight':30s} {'true':>7} {'point_est':>10} {'lb':>7} {'std':>7} {'gap':>7}")
    for r in passing[:15]:
        print(f"{r['idx']:5d} {r['flight']:30s} {r['true_rul']:7.1f} {r['point_est']:10.1f} "
              f"{r['lb']:7.1f} {r['std']:7.1f} {r['gap']:7.1f}")

    # -----------------------------------------------------------------
    # Calibration-vs-model diagnosis
    # -----------------------------------------------------------------
    print("\n=== Diagnosis: is this calibration (band too narrow) or model accuracy? ===")
    point_est_errors = [r["point_est"] - r["true_rul"] for r in failing]
    lb_overshoots = [r["lb"] - r["true_rul"] for r in failing]  # how far lb is ABOVE true (should be <=0)

    print(f"  Failing windows' point-estimate error (point_est - true_rul):")
    print(f"    mean={np.mean(point_est_errors):+.2f}  min={min(point_est_errors):+.2f}  "
          f"max={max(point_est_errors):+.2f}")
    print(f"  Failing windows' lower-bound overshoot (lb - true_rul, positive = lb wrongly above true):")
    print(f"    mean={np.mean(lb_overshoots):+.2f}  min={min(lb_overshoots):+.2f}  "
          f"max={max(lb_overshoots):+.2f}")

    # If point_est itself is already above true_rul for most failing windows,
    # no amount of band-widening on a reasonable point_est fixes it -- that's
    # a model-accuracy problem, same category as the original bucket-0 issue.
    point_est_also_over = sum(1 for e in point_est_errors if e > 0)
    print(f"\n  Point estimate ALSO overshoots true_rul in {point_est_also_over}/{len(failing)} "
          f"failing windows.")
    print("  If that count is most/all of them: point_est itself is biased high here, and")
    print("  widening the conformal band alone won't fix it -- this needs the same kind of")
    print("  investigation bucket-0 got (OOD check, feature separation, possible reweighting),")
    print("  not just a calibration refit.")
    print("  If that count is few/none: point_est is roughly on target and the band is simply")
    print("  too narrow for this bucket -- widening/refitting calibration on this bucket alone")
    print("  is the cheap, low-risk fix to try first.")

    # -----------------------------------------------------------------
    # Per-flight / per-base-trajectory breakdown
    # -----------------------------------------------------------------
    print("\n=== Failing-window count by flight (checking for collapse on 1-2 flights) ===")
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

    print("\nSTOP-OR-CONTINUE: if failures cluster on 1-2 flights/bases, this may be a")
    print("per-trajectory quirk rather than a general bucket problem -- worth inspecting")
    print("those specific flights' raw traces before deciding on a fix approach. If spread")
    print("evenly, it's a general bucket-level issue and the calibration-vs-model split")
    print("above is the deciding factor for which fix to pursue.")


if __name__ == "__main__":
    main()