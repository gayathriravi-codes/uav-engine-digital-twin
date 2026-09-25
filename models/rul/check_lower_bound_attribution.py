"""
check_lower_bound_attribution.py

v10 follow-up -- before deciding whether to re-seed variant 5, check whether
its badly-low predictions on base 008/bucket(100,200) are currently SETTING
predict_rul_ensemble()'s rul_lower_bound (which is MIN across variants,
clipped >= 0, before calibration's conformal band is applied).

This matters because:
  - If variant 5 is usually the MIN on these windows, removing/re-seeding it
    will likely RAISE the raw lower bound on base 008's bucket(100,200)
    windows -- possibly a good thing (less falsely-pessimistic), but also
    possibly removing an accidental safety margin if the calibration's
    conformal band was implicitly relying on that low value for coverage.
  - If variant 5 is rarely/never the MIN (some other variant already
    predicts lower), then it's mostly just dragging the POINT ESTIMATE
    (mean) down and barely affects the lower bound at all -- re-seeding it
    would mainly fix the point estimate's bias, with limited effect on
    already-computed coverage numbers.

Either way, THIS SCRIPT DOES NOT ITSELF DECIDE WHAT TO DO -- it just
establishes the mechanism before you retrain anything, so the effect of
re-seeding variant 5 is predictable rather than a surprise discovered only
after refitting calibration.

Uses the already-trained, already-saved ensemble via load_ensemble() --
does not retrain anything. Val-only. Does NOT touch test.

Run: python models\\rul\\check_lower_bound_attribution.py
"""
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rul import load_all_flights, DEVICE
from train_rul_ensemble import (
    base_trajectory_id,
    build_windowed_dataset_v6,
    load_ensemble,
    SEEDS, DROPOUTS, HIDDEN_SIZES,
)
from schema import SENSOR_FIELDS
from sklearn.model_selection import train_test_split

BUCKET_LO, BUCKET_HI = 100, 200
TARGET_BASE = "008"
COMPARISON_BASE = "005"
VARIANT_OF_INTEREST = 5


def predict_single_variant(model, scaler, dropped_idx, window):
    scaled = scaler.transform(window.reshape(-1, window.shape[-1])).reshape(window.shape)
    window_variant = scaled.copy()
    if len(dropped_idx) > 0:
        window_variant[:, dropped_idx] = 0.0
    model.eval()
    with torch.no_grad():
        x = torch.tensor(window_variant, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        return model(x).item()


def main():
    print("Loading flights ...")
    flights = load_all_flights()

    print("Building v10 (11-feature) windowed dataset ...")
    X, y, flight_ids = build_windowed_dataset_v6(flights)

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
    val_bases_arr = np.array([base_trajectory_id(f) for f in val_flight_ids])

    bucket_mask = (y_val >= BUCKET_LO) & (y_val < BUCKET_HI)

    print("\nLoading the already-trained, already-saved ensemble (no retraining) ...")
    models, scaler, dropped_idx_list = load_ensemble()
    n_variants = len(models)

    for target_base, label in [(TARGET_BASE, "TARGET (known-bad)"), (COMPARISON_BASE, "COMPARISON (healthy)")]:
        idx = np.where(bucket_mask & (val_bases_arr == target_base))[0]
        true_vals = y_val[idx]

        print("\n" + "=" * 90)
        print(f"BASE {target_base} -- {label} -- n={len(idx)} windows in bucket({BUCKET_LO},{BUCKET_HI})")
        print("=" * 90)

        # per-window, per-variant raw predictions
        all_variant_preds = np.zeros((n_variants, len(idx)))
        for v_idx, (model, dropped_idx) in enumerate(zip(models, dropped_idx_list)):
            for j, i in enumerate(idx):
                all_variant_preds[v_idx, j] = predict_single_variant(model, scaler, dropped_idx, X_val[i])

        # For each window: which variant produced the MIN (pre-clip) prediction?
        min_variant_per_window = np.argmin(all_variant_preds, axis=0)
        min_values = np.min(all_variant_preds, axis=0)
        min_values_clipped = np.maximum(0.0, min_values)

        counts = Counter(min_variant_per_window.tolist())
        print(f"\n  How often each variant sets the raw lower_bound (MIN across variants), n={len(idx)} windows:")
        for v_idx in range(n_variants):
            n_times = counts.get(v_idx, 0)
            pct = 100.0 * n_times / len(idx)
            marker = "  <-- VARIANT OF INTEREST" if v_idx == VARIANT_OF_INTEREST else ""
            print(f"    variant {v_idx} (seed={SEEDS[v_idx]}, hidden={HIDDEN_SIZES[v_idx]}): "
                  f"{n_times:3d}/{len(idx)} windows ({pct:5.1f}%){marker}")

        v5_sets_bound_pct = 100.0 * counts.get(VARIANT_OF_INTEREST, 0) / len(idx)

        # What would the lower bound look like WITHOUT variant 5 at all?
        other_idx = [v for v in range(n_variants) if v != VARIANT_OF_INTEREST]
        min_without_v5 = np.min(all_variant_preds[other_idx, :], axis=0)
        min_without_v5_clipped = np.maximum(0.0, min_without_v5)

        mean_lb_with_v5 = np.mean(min_values_clipped)
        mean_lb_without_v5 = np.mean(min_without_v5_clipped)
        shift = mean_lb_without_v5 - mean_lb_with_v5

        # Coverage-relevant: does removing v5 change whether lower_bound <= true_rul
        # (a basic sanity property -- lower_bound should stay <= true for "coverage"
        # in the conformal sense used elsewhere in this project)
        coverage_with_v5 = np.mean(min_values_clipped <= true_vals) * 100
        coverage_without_v5 = np.mean(min_without_v5_clipped <= true_vals) * 100

        print(f"\n  Variant 5 sets the raw lower_bound on {v5_sets_bound_pct:.1f}% of these windows.")
        print(f"  Mean raw lower_bound WITH variant 5:    {mean_lb_with_v5:7.2f}")
        print(f"  Mean raw lower_bound WITHOUT variant 5: {mean_lb_without_v5:7.2f}  (shift: {shift:+.2f})")
        print(f"  Raw lower_bound <= true_rul (with v5):    {coverage_with_v5:5.1f}%")
        print(f"  Raw lower_bound <= true_rul (without v5): {coverage_without_v5:5.1f}%")

    print("\n" + "=" * 90)
    print("INTERPRETATION")
    print("=" * 90)
    print("If variant 5 sets the lower_bound on a LARGE fraction of base 008's bucket(100,200)")
    print("windows, AND removing it shifts the mean lower_bound UP noticeably, then variant 5's")
    print("badly-low predictions are currently providing (accidental) conservative coverage --")
    print("re-seeding it could RAISE the lower bound and potentially reduce coverage on this")
    print("specific slice unless calibration is refit and re-verified afterward.")
    print()
    print("If variant 5 rarely/never sets the lower_bound here (some other variant is already")
    print("lower), then variant 5 is mainly dragging the MEAN (point estimate) down via its bias,")
    print("with little effect on the lower bound -- re-seeding it would mostly fix the point")
    print("estimate's accuracy without materially changing today's coverage numbers.")
    print()
    print("Either way: if you do re-seed variant 5, refit calibration afterward and re-run")
    print("check_ensemble_bucket100_200_breakdown.py before considering this settled -- do not")
    print("assume the effect predicted here transfers exactly once calibration is refit on top.")


if __name__ == "__main__":
    main()