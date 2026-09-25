"""
check_coverage_on_val.py -- ONE-TIME-per-debug-cycle sanity check of the
v8 OOD-aware calibration fix, run on VALIDATION data (not test).

Purpose: confirm (a) the is_ood flag is actually firing for a nonzero
number of windows, (b) bucket (0,50)'s coverage responds, and (c) WHY
any remaining bucket-(0,50) failures are still failing -- before spending
another pull on the real held-out test set via run_coverage_check.py.

Run: python check_coverage_on_val.py
"""
import os
import sys

import numpy as np
import torch
from sklearn.model_selection import train_test_split
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rul import load_all_flights, MODEL_OUT_DIR
from train_rul_ensemble import (
    RULRegressorVariant, build_windowed_dataset_v5, HIDDEN_SIZES, N_VARIANTS,
    compute_ood_zscore_distance, predict_rul_ensemble,
)
from calibrate_rul import load_calibration_params, check_per_bucket_coverage, apply_calibration

OOD_THRESHOLD = 2.0
BUCKETS = [(0, 50), (50, 100), (100, 200), (200, 350)]


def bucket_for(true_rul):
    for lo, hi in BUCKETS:
        if lo <= true_rul < hi:
            return (lo, hi)
    return None


if __name__ == "__main__":
    flights = load_all_flights()
    X, y, flight_ids = build_windowed_dataset_v5(flights)

    unique_flights = np.unique(flight_ids)
    train_flights, temp_flights = train_test_split(unique_flights, test_size=0.3, random_state=42)
    val_flights, test_flights = train_test_split(temp_flights, test_size=0.5, random_state=42)
    val_mask = np.isin(flight_ids, val_flights)
    X_val, y_val = X[val_mask], y[val_mask]
    print(f"Validation windows: {len(X_val)} (val set, NOT test -- safe to re-run)")

    scaler = joblib.load(os.path.join(MODEL_OUT_DIR, "rul_ensemble_scaler.joblib"))
    dropped_idx_list = joblib.load(os.path.join(MODEL_OUT_DIR, "rul_ensemble_dropped_idx.joblib"))
    models = []
    for i in range(N_VARIANTS):
        model = RULRegressorVariant(hidden_size=HIDDEN_SIZES[i])
        model.load_state_dict(torch.load(os.path.join(MODEL_OUT_DIR, f"rul_model_variant_{i}.pt")))
        model.eval()
        models.append(model)

    # --- Step 1: global + per-bucket OOD flag counts ---
    ood_flags = np.zeros(len(X_val), dtype=bool)
    for i in range(len(X_val)):
        d = compute_ood_zscore_distance(X_val[i], scaler)
        ood_flags[i] = d > OOD_THRESHOLD

    print(f"\nWindows flagged OOD (z-score distance > {OOD_THRESHOLD}): {ood_flags.sum()} / {len(X_val)}")
    if ood_flags.sum() == 0:
        print("WARNING: zero windows flagged OOD -- the fix cannot change anything.")
        print("Check ood_zscore_threshold, or whether compute_ood_zscore_distance")
        print("is being fed correctly-scaled/shaped windows.")

    print("\nPer-bucket OOD flag counts:")
    bucket_indices = {b: [] for b in BUCKETS}
    for i, true_rul in enumerate(y_val):
        b = bucket_for(true_rul)
        if b is not None:
            bucket_indices[b].append(i)
    for b in BUCKETS:
        idxs = bucket_indices[b]
        n_flagged = sum(ood_flags[i] for i in idxs)
        print(f"  bucket {b}: {n_flagged} / {len(idxs)} windows flagged OOD")

    # --- Step 2: overall per-bucket coverage (existing check) ---
    params = load_calibration_params()
    print("\nRunning check_per_bucket_coverage() on VALIDATION set (with v8 OOD fix active) ...")
    check_per_bucket_coverage(X_val, y_val, models, scaler, dropped_idx_list, params)

    # --- Step 3: per-window diagnosis for bucket (0,50) only ---
    print("\n" + "=" * 70)
    print("Per-window diagnosis for bucket (0, 50)")
    print("=" * 70)
    target_bucket = (0, 50)
    idxs = bucket_indices[target_bucket]

    still_failing = []
    for i in idxs:
        true_rul = y_val[i]
        point_est = predict_rul_ensemble(
            X_val[i], models, scaler, dropped_idx_list, calibrated=False
        )["point_estimate_timesteps"]
        is_ood = bool(ood_flags[i])
        y_cal, lb_cal = apply_calibration(point_est, params, is_ood=is_ood)
        covered = true_rul >= lb_cal  # one-sided, matching check_per_bucket_coverage
        if not covered:
            still_failing.append({
                "idx": i,
                "true_rul": true_rul,
                "point_est": point_est,
                "is_ood": is_ood,
                "y_cal": y_cal,
                "lb_cal": lb_cal,
            })

    n_total = len(idxs)
    n_failing = len(still_failing)
    print(f"\n{n_failing} / {n_total} bucket-(0,50) windows still failing coverage post-v8-fix.\n")

    if n_failing == 0:
        print("All bucket-(0,50) windows covered on validation -- fix looks sufficient here.")
    else:
        # Note: a window with is_ood=True always has lb_cal forced to 0.0,
        # and true_rul is always >= 0, so an OOD-flagged window can never
        # fail this coverage check (true_rul >= lb_cal is guaranteed).
        # Every remaining failure is therefore a window the OOD detector
        # did NOT flag -- confirm that below rather than assume it.
        flagged_but_failing = [w for w in still_failing if w["is_ood"]]
        not_flagged_failing = [w for w in still_failing if not w["is_ood"]]

        if flagged_but_failing:
            print("UNEXPECTED: some OOD-flagged windows still failed coverage -- this should be "
                  "impossible (lb should be 0.0). Check apply_calibration's is_ood handling:")
            for w in flagged_but_failing:
                print(f"  idx={w['idx']:>5}  true={w['true_rul']:.1f}  point_est={w['point_est']:.1f}  "
                      f"y_cal={w['y_cal']:.1f}  lb={w['lb_cal']:.1f}")

        print(f"\nAll {len(not_flagged_failing)} failing windows are NOT OOD-flagged "
              f"(detector missed them -- a threshold/scaling issue, not an lb-override issue):")
        for w in not_flagged_failing:
            d = compute_ood_zscore_distance(X_val[w["idx"]], scaler)
            print(f"  idx={w['idx']:>5}  true={w['true_rul']:.1f}  point_est={w['point_est']:.1f}  "
                  f"y_cal={w['y_cal']:.1f}  lb={w['lb_cal']:.1f}  ood_distance={d:.3f}  "
                  f"(threshold={OOD_THRESHOLD})")

        # --- Step 4: is ensemble disagreement (std) a better signal than z-distance? ---
        print("\n" + "=" * 70)
        print("Ensemble std comparison: failing windows vs. a covered sample")
        print("=" * 70)

        def ensemble_std(i):
            scaled = scaler.transform(X_val[i].reshape(-1, X_val[i].shape[-1])).reshape(X_val[i].shape)
            preds = []
            with torch.no_grad():
                for model, dropped_idx in zip(models, dropped_idx_list):
                    model.eval()
                    wv = scaled.copy()
                    if len(dropped_idx) > 0:
                        wv[:, dropped_idx] = 0.0
                    x = torch.tensor(wv, dtype=torch.float32).unsqueeze(0)
                    preds.append(model(x).item())
            return float(np.std(preds))

        failing_idxs = [w["idx"] for w in not_flagged_failing]
        failing_stds = [ensemble_std(i) for i in failing_idxs]

        covered_idxs_in_bucket = [i for i in idxs if i not in failing_idxs]
        rng = np.random.RandomState(0)
        sample_covered = rng.choice(covered_idxs_in_bucket,
                                     size=min(20, len(covered_idxs_in_bucket)), replace=False)
        covered_stds = [ensemble_std(i) for i in sample_covered]

        print(f"\nFailing windows (n={len(failing_stds)}): "
              f"mean std={np.mean(failing_stds):.2f}  min={np.min(failing_stds):.2f}  "
              f"max={np.max(failing_stds):.2f}")
        print(f"Covered sample (n={len(covered_stds)}):  "
              f"mean std={np.mean(covered_stds):.2f}  min={np.min(covered_stds):.2f}  "
              f"max={np.max(covered_stds):.2f}")

        if np.mean(failing_stds) > 1.5 * np.mean(covered_stds):
            print("\n-> Failing windows show notably HIGHER ensemble disagreement. std could be a")
            print("   usable second OOD signal (e.g. flag is_ood if z_distance > thr OR std > X).")
        else:
            print("\n-> No clear separation in ensemble std either. The 5 variants agree with each")
            print("   other AND with the feature-space 'normal' region -- they're just all")
            print("   confidently wrong together on these near-end-of-life windows. This looks")
            print("   like a genuine upstream model gap (doc section 4), not something an OOD")
            print("   gate of any kind (feature-distance or disagreement-based) can catch --")
            print("   worth documenting as a known limitation rather than continuing to tune")
            print("   calibration/OOD parameters.")

        if not_flagged_failing:
            print("\nSuggestion: print these windows' actual z-score distances (they're <= "
                  f"{OOD_THRESHOLD} but may be close) to see if lowering ood_zscore_threshold a")
            print("bit would catch them without flagging much of the rest of validation as OOD.")