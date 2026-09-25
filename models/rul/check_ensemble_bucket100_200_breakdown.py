"""
check_ensemble_bucket100_200_breakdown.py

v10 follow-up check -- BEFORE trusting the retrained 7-variant ensemble's
calibration for bucket(100,200) and before touching run_coverage_check.py
/ test.

Context: train_rul_ensemble.py's own val-only spot-check showed
bucket(100,200) RMSE=8.07 / max_overshoot=+1.04 -- much better than the
pre-elapsed_fraction baseline (43.93 / +52.39), but notably worse than
the single-variant experiment's isolated result (2.47 / +0.12). The
Stage 0 calibration bias for this bucket (-15.13) is also 2-14x larger
in magnitude than every other bucket's, and its Stage 1 band (18.74) is
the widest of the four.

This does NOT by itself tell us whether that degradation is:
  (a) uniform dilution -- expected, since 6 of 7 variants don't have
      elapsed_fraction's signal isolated the way the single-variant
      experiment did (they're also doing feature bagging / different
      hidden sizes / different dropout), or
  (b) concentrated on specific flights/bases -- which would suggest a
      real problem (e.g. one variant reintroducing a version of the old
      plateau behavior) rather than expected ensemble averaging.

Same discipline as check_elapsed_fraction_stability.py and the earlier
reweighting-experiment lesson: don't trust an aggregate number without
a per-flight/per-base breakdown.

Uses the ALREADY-TRAINED, ALREADY-SAVED ensemble via load_ensemble() --
does not retrain anything. Val-only. Does NOT touch test.

Run: python models\\rul\\check_ensemble_bucket100_200_breakdown.py
"""
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rul import load_all_flights, DEVICE
from train_rul_ensemble import (
    base_trajectory_id,
    build_windowed_dataset_v6,
    load_ensemble,
    predict_rul_ensemble,
)
from sklearn.model_selection import train_test_split

BUCKET_LO, BUCKET_HI = 100, 200


def main():
    print("Loading flights ...")
    flights = load_all_flights()

    print("Building v10 (11-feature) windowed dataset -- must match what the saved ensemble was trained on ...")
    X, y, flight_ids = build_windowed_dataset_v6(flights)
    print(f"Total windows: {len(X)}  Feature columns: {X.shape[-1]}")

    # Identical split logic/seeds to train_rul_ensemble.py's __main__ --
    # imported base_trajectory_id, same random_state, same order of
    # operations, so this reproduces the exact same val set the ensemble
    # was actually evaluated against during training.
    unique_flights = np.unique(flight_ids)
    base_ids = np.array([base_trajectory_id(f) for f in unique_flights])
    unique_bases = np.unique(base_ids)

    train_bases, temp_bases = train_test_split(unique_bases, test_size=0.3, random_state=42)
    val_bases, test_bases = train_test_split(temp_bases, test_size=0.5, random_state=42)
    val_bases_set = set(val_bases)

    val_flights = np.array([f for f in unique_flights if base_trajectory_id(f) in val_bases_set])
    val_mask = np.isin(flight_ids, val_flights)

    X_val, y_val = X[val_mask], y[val_mask]
    val_flight_ids = flight_ids[val_mask]

    bucket_mask = (y_val >= BUCKET_LO) & (y_val < BUCKET_HI)
    print(f"\nVal windows in bucket ({BUCKET_LO},{BUCKET_HI}): {bucket_mask.sum()}")
    print("(Should be 360, matching train_rul_ensemble.py's own spot-check n=360.)")

    print("\nLoading the already-trained, already-saved ensemble (no retraining) ...")
    models, scaler, dropped_idx_list = load_ensemble()

    print("\nRunning predict_rul_ensemble() on bucket(100,200) val windows (calibrated=False, raw ensemble output --")
    print("matches what train_rul_ensemble.py's own spot-check used) ...")
    bucket_idx = np.where(bucket_mask)[0]
    preds = np.zeros(len(bucket_idx))
    for j, i in enumerate(bucket_idx):
        result = predict_rul_ensemble(X_val[i], models, scaler, dropped_idx_list, calibrated=False)
        preds[j] = result["point_estimate_timesteps"]

    true_vals = y_val[bucket_idx]
    overall_rmse = np.sqrt(np.mean((preds - true_vals) ** 2))
    overall_max_overshoot = np.max(preds - true_vals)
    overall_bias = np.mean(preds - true_vals)
    print(f"\nOverall bucket({BUCKET_LO},{BUCKET_HI}) check: n={len(bucket_idx)}  rmse={overall_rmse:.2f}  "
          f"max_overshoot={overall_max_overshoot:+.2f}  bias={overall_bias:+.2f}")
    print("(Reference: train_rul_ensemble.py's own spot-check reported RMSE=8.07, max_overshoot=+1.04 --")
    print(" should match closely; any large discrepancy here means this script's split/eval doesn't")
    print(" actually reproduce that check, and that mismatch needs resolving before trusting anything below.)")

    print("\n" + "=" * 78)
    print("PER-FLIGHT BREAKDOWN, within bucket (100,200)")
    print("=" * 78)
    by_flight = defaultdict(lambda: {"true": [], "pred": []})
    for j, i in enumerate(bucket_idx):
        fid = val_flight_ids[i]
        by_flight[fid]["true"].append(true_vals[j])
        by_flight[fid]["pred"].append(preds[j])

    flight_rmses = []
    for fid, d in sorted(by_flight.items()):
        true_arr = np.array(d["true"])
        pred_arr = np.array(d["pred"])
        rmse = np.sqrt(np.mean((pred_arr - true_arr) ** 2))
        max_over = np.max(pred_arr - true_arr)
        bias = np.mean(pred_arr - true_arr)
        flight_rmses.append(rmse)
        print(f"  {fid:30s}: n={len(true_arr):3d}  rmse={rmse:6.2f}  max_overshoot={max_over:+7.2f}  bias={bias:+7.2f}")

    print(f"\n  Per-flight RMSE spread: min={min(flight_rmses):.2f}  max={max(flight_rmses):.2f}  "
          f"spread={max(flight_rmses)-min(flight_rmses):.2f}")

    print("\n" + "=" * 78)
    print("PER-BASE-TRAJECTORY BREAKDOWN, within bucket (100,200)")
    print("=" * 78)
    by_base = defaultdict(lambda: {"true": [], "pred": []})
    for j, i in enumerate(bucket_idx):
        base = base_trajectory_id(val_flight_ids[i])
        by_base[base]["true"].append(true_vals[j])
        by_base[base]["pred"].append(preds[j])

    base_rmses = {}
    for base, d in sorted(by_base.items()):
        true_arr = np.array(d["true"])
        pred_arr = np.array(d["pred"])
        rmse = np.sqrt(np.mean((pred_arr - true_arr) ** 2))
        max_over = np.max(pred_arr - true_arr)
        bias = np.mean(pred_arr - true_arr)
        base_rmses[base] = rmse
        print(f"  base {base}: n={len(true_arr):3d}  rmse={rmse:6.2f}  max_overshoot={max_over:+7.2f}  bias={bias:+7.2f}")

    print("\n-- Base 005's previously-known anomalous windows (true_rul~165, was flat ~210-214") 
    print("   pre-elapsed_fraction; resolved to ~164.1 in the single-variant check) --")
    base005_idx = [
        (j, i) for j, i in enumerate(bucket_idx)
        if base_trajectory_id(val_flight_ids[i]) == "005" and abs(true_vals[j] - 165.0) < 1e-6
    ]
    if base005_idx:
        for j, i in base005_idx[:6]:
            print(f"  {val_flight_ids[i]:30s}  true={true_vals[j]:.1f}  pred={preds[j]:.1f}  error={preds[j]-true_vals[j]:+.1f}")
    else:
        print("  (no matching windows found -- base 005 may not be in this val split, or true_rul~165")
        print("   doesn't appear at this bucket boundary for this base; check base_rmses above for base 005 directly)")

    print("\n" + "=" * 78)
    print("STOP-OR-CONTINUE")
    print("=" * 78)
    rmse_spread = max(flight_rmses) - min(flight_rmses)
    base_rmse_values = list(base_rmses.values())
    base_spread = max(base_rmse_values) - min(base_rmse_values) if len(base_rmse_values) > 1 else 0.0
    print(f"Per-flight RMSE spread: {rmse_spread:.2f}   Per-base RMSE spread: {base_spread:.2f}")
    print("If RMSE is tightly clustered across flights/bases (similar to the single-variant check's")
    print("1.66-3.83 band), the ensemble-level degradation vs. the single-variant experiment is")
    print("uniform dilution from the other 6 variants' diversity -- expected, not a red flag, safe to")
    print("proceed with the current calibration.")
    print("If one or two flights/bases are dramatically worse than the rest (e.g. one base showing")
    print("10x the RMSE of the others), that's concentrated failure, not dilution -- the same failure")
    print("shape as the original reweighting experiment. Do NOT proceed to run_coverage_check.py/test")
    print("with the current calibration until that specific flight/base is investigated -- it likely")
    print("means one variant (or a specific feature-bagging combination) is reintroducing a version of")
    print("the pre-fix plateau behavior for that trajectory shape specifically.")


if __name__ == "__main__":
    main()