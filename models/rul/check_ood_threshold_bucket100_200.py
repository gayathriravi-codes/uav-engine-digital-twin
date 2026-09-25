"""
check_ood_threshold_bucket100_200.py

Does compute_ood_zscore_distance() already separate the known-anomalous
"plateau" windows (val's base-005 anomaly; test's base-000/013 failures)
from normal bucket-(100,200) windows -- just at a threshold higher than
the current ood_zscore_threshold=2.0 -- or does it not separate them at
all (in which case tightening the threshold is a dead end, same as the
std-gate rejected earlier in this investigation)?

Val-only for the sweep itself (base-005's anomaly is a known val example).
Does NOT touch test -- test's failing windows are only used here as
ALREADY-KNOWN data from the prior diagnostic run's printed output, not
re-fetched by running anything against test again.

Run: python models\\rul\\check_ood_threshold_bucket100_200.py
"""
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rul import load_all_flights
from train_rul_ensemble import (
    build_windowed_dataset_v5, load_ensemble, predict_rul_ensemble,
    compute_ood_zscore_distance, base_trajectory_id,
)

BUCKET_LO, BUCKET_HI = 100, 200
CANDIDATE_THRESHOLDS = [2.0, 1.8, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0]


def main():
    print("Loading flights and building windowed dataset ...")
    flights = load_all_flights()
    X, y, flight_ids = build_windowed_dataset_v5(flights)

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

    models, scaler, dropped_idx_list = load_ensemble()

    # Isolate bucket (100,200) val windows, compute point_est + ood_distance for each.
    records = []
    for i in range(len(X_val)):
        true_rul = y_val[i]
        if not (BUCKET_LO <= true_rul < BUCKET_HI):
            continue
        result = predict_rul_ensemble(X_val[i], models, scaler, dropped_idx_list, calibrated=False)
        point_est = result["point_estimate_timesteps"]
        ood_dist = compute_ood_zscore_distance(X_val[i], scaler)
        base = base_trajectory_id(val_flight_ids_arr[i])
        records.append({
            "idx": i, "base": base, "true_rul": true_rul,
            "point_est": point_est, "ood_dist": ood_dist,
            "overshoot": point_est - true_rul,
        })

    # "Anomalous" = same signature as the known base-005 failures: point_est
    # overshoots true_rul by a lot (>30, well above normal noise in this bucket).
    anomalous = [r for r in records if r["overshoot"] > 30]
    normal = [r for r in records if r["overshoot"] <= 30]

    print(f"Bucket ({BUCKET_LO},{BUCKET_HI}) val windows: {len(records)}  "
          f"anomalous(overshoot>30): {len(anomalous)}  normal: {len(normal)}\n")

    print("=== Anomalous windows: base, true_rul, point_est, overshoot, ood_dist ===")
    for r in sorted(anomalous, key=lambda r: -r["overshoot"]):
        print(f"  base={r['base']}  true={r['true_rul']:6.1f}  point_est={r['point_est']:7.1f}  "
              f"overshoot={r['overshoot']:6.1f}  ood_dist={r['ood_dist']:.3f}")

    print(f"\n  anomalous ood_dist: min={min(r['ood_dist'] for r in anomalous):.3f}  "
          f"max={max(r['ood_dist'] for r in anomalous):.3f}  "
          f"mean={np.mean([r['ood_dist'] for r in anomalous]):.3f}")
    print(f"  normal    ood_dist: min={min(r['ood_dist'] for r in normal):.3f}  "
          f"max={max(r['ood_dist'] for r in normal):.3f}  "
          f"mean={np.mean([r['ood_dist'] for r in normal]):.3f}")

    print("\n=== Threshold sweep: would lowering ood_zscore_threshold catch the anomalous")
    print("    windows WITHOUT flagging most of the normal ones? ===")
    for thresh in CANDIDATE_THRESHOLDS:
        caught_anom = sum(1 for r in anomalous if r["ood_dist"] > thresh)
        flagged_normal = sum(1 for r in normal if r["ood_dist"] > thresh)
        print(f"  threshold={thresh:.1f}:  anomalous caught={caught_anom}/{len(anomalous)}  "
              f"normal falsely flagged={flagged_normal}/{len(normal)} "
              f"({100*flagged_normal/len(normal):.1f}%)")

    print("\nSTOP-OR-CONTINUE: a usable threshold needs HIGH anomalous-catch and LOW")
    print("normal-false-flag (ideally <5-10%, since this bucket already passes at 98%+")
    print("on val -- a threshold that flags many normal windows will manufacture a new")
    print("coverage failure here to fix a different one, same failure mode as the")
    print("rejected std-gate). If no threshold in this list achieves that, ood_zscore")
    print("distance does not separate this failure mode -- tightening it is a dead end,")
    print("and the real fix needs a different signal (e.g. a trajectory-shape feature),")
    print("not a threshold tweak.")


if __name__ == "__main__":
    main()