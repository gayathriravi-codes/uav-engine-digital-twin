"""
check_variant1_lower_bound_attribution.py

v10 follow-up -- mirrors check_lower_bound_attribution.py's original
question about variant 5, now applied to variant 1 (the new worst variant
on base 008/bucket(100,200) per check_base008_per_variant.py, after
variant 5's re-seed fixed the original outlier).

Question: is variant 1 disproportionately setting the raw ensemble
lower_bound (MIN across variants) on base 008's bucket(100,200) windows --
well above its "fair share" -- the way variant 5 was (30.8% vs. a healthy
5.0% on base 005)? If so, re-seeding variant 1 should meaningfully raise
the mean lower_bound on base 008 specifically, the same low-risk shape as
the variant 5 fix. Also checks whether raw lower_bound <= true_rul
coverage is preserved with and without variant 1 -- variant 5's fix had
100% coverage either way; that is NOT guaranteed to hold again here and
must be checked, not assumed.

Built directly on check_base008_per_variant.py's verified split-reproduction
and per-variant prediction logic -- NOT reimplemented independently, per
standing project discipline.

Uses the already-trained, already-saved ensemble via load_ensemble() --
does not retrain anything. Val-only. Does NOT touch test.

Run: python models\\rul\\check_variant1_lower_bound_attribution.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rul import load_all_flights, DEVICE
from train_rul_ensemble import (
    base_trajectory_id,
    build_windowed_dataset_v6,
    load_ensemble,
)
from schema import SENSOR_FIELDS
from sklearn.model_selection import train_test_split

BUCKET_LO, BUCKET_HI = 100, 200
TARGET_VARIANT = 1
TARGET_BASE = "008"
COMPARISON_BASE = "005"


def predict_single_variant(model, scaler, dropped_idx, window):
    """Same as check_base008_per_variant.py."""
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

    # Identical split to train_rul_ensemble.py / check_base008_per_variant.py
    unique_flights = np.unique(flight_ids)
    base_ids_unique = np.array([base_trajectory_id(f) for f in unique_flights])
    unique_bases = np.unique(base_ids_unique)

    train_bases, temp_bases = train_test_split(unique_bases, test_size=0.3, random_state=42)
    val_bases, test_bases = train_test_split(temp_bases, test_size=0.5, random_state=42)
    val_bases_set = set(val_bases)

    val_flights = np.array([f for f in unique_flights if base_trajectory_id(f) in val_bases_set])
    val_mask = np.isin(flight_ids, val_flights)

    X_val, y_val = X[val_mask], y[val_mask]
    val_flight_ids = flight_ids[val_mask]

    bucket_mask = (y_val >= BUCKET_LO) & (y_val < BUCKET_HI)

    print("\nLoading the already-trained, already-saved ensemble (no retraining) ...")
    models, scaler, dropped_idx_list = load_ensemble()
    n_variants = len(models)
    other_variants = [v for v in range(n_variants) if v != TARGET_VARIANT]

    def analyze_base(target_base):
        base_idx = np.array([
            i for i in np.where(bucket_mask)[0]
            if base_trajectory_id(val_flight_ids[i]) == target_base
        ])
        n = len(base_idx)
        if n == 0:
            print(f"  base {target_base}: no windows in bucket({BUCKET_LO},{BUCKET_HI}), skipping")
            return None

        true_vals = y_val[base_idx]

        # Per-variant predictions for every window in this slice
        all_preds = np.zeros((n_variants, n))
        for v in range(n_variants):
            all_preds[v] = np.array([
                predict_single_variant(models[v], scaler, dropped_idx_list[v], X_val[i])
                for i in base_idx
            ])

        lower_bound_with = all_preds.min(axis=0)
        argmin_with = all_preds.argmin(axis=0)
        share_target_is_min = float(np.mean(argmin_with == TARGET_VARIANT))

        lower_bound_without = all_preds[other_variants].min(axis=0)
        mean_shift = float(np.mean(lower_bound_without - lower_bound_with))

        coverage_with = float(np.mean(lower_bound_with <= true_vals))
        coverage_without = float(np.mean(lower_bound_without <= true_vals))

        fair_share = 1.0 / n_variants
        print(f"  base {target_base} (n={n}):")
        print(f"    variant {TARGET_VARIANT} sets raw lower_bound on "
              f"{share_target_is_min*100:.1f}% of windows "
              f"(fair share would be ~{fair_share*100:.1f}%)")
        print(f"    removing variant {TARGET_VARIANT} shifts mean lower_bound "
              f"by {mean_shift:+.2f}")
        print(f"    coverage (lower_bound <= true_rul): "
              f"with={coverage_with*100:.1f}%  without={coverage_without*100:.1f}%")
        return dict(share=share_target_is_min, shift=mean_shift,
                    cov_with=coverage_with, cov_without=coverage_without)

    print("\n" + "=" * 88)
    print(f"VARIANT {TARGET_VARIANT} LOWER-BOUND ATTRIBUTION -- bucket({BUCKET_LO},{BUCKET_HI})")
    print("=" * 88)
    print(f"\nTarget (known-outlier-on-this-slice) base {TARGET_BASE}:")
    target_result = analyze_base(TARGET_BASE)

    print(f"\nComparison (healthy) base {COMPARISON_BASE}:")
    comparison_result = analyze_base(COMPARISON_BASE)

    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    if target_result and comparison_result:
        t, c = target_result, comparison_result

        print(f"Variant {TARGET_VARIANT} sets the min on base {TARGET_BASE} "
              f"{t['share']*100:.1f}% of the time vs. {c['share']*100:.1f}% on base "
              f"{COMPARISON_BASE}.")

        if t['cov_with'] < 1.0 - 1e-9 or c['cov_with'] < 1.0 - 1e-9:
            print("WARNING: current coverage with variant 1 included is NOT 100% on "
                  "one of these bases -- re-check before assuming coverage is safe "
                  "either way (this differed from variant 5's case, where coverage "
                  "was 100% with and without).")
        elif t['cov_without'] < 1.0 - 1e-9 or c['cov_without'] < 1.0 - 1e-9:
            print("WARNING: removing/re-seeding variant 1 would DROP coverage below "
                  "100% on one of these bases. Unlike variant 5's case, this is NOT a "
                  "free fix -- re-seeding could introduce a coverage risk that needs "
                  "investigating before proceeding.")
        else:
            print("Coverage (lower_bound <= true_rul) is 100% both with and without "
                  "variant 1 on both bases -- consistent with variant 5's case: no "
                  "coverage risk from re-seeding.")

        if t['share'] > 2 * c['share'] and t['shift'] > c['shift']:
            print(f"\nVariant {TARGET_VARIANT} is disproportionately dragging down base "
                  f"{TARGET_BASE}'s lower_bound specifically (shift {t['shift']:+.2f} "
                  f"vs. {c['shift']:+.2f} on the healthy base) -- same shape as variant "
                  "5's original problem. Re-seeding is reasonable on the same rationale "
                  "used for variant 5.")
        else:
            print(f"\nVariant {TARGET_VARIANT}'s lower-bound dominance on base "
                  f"{TARGET_BASE} is not clearly disproportionate vs. base "
                  f"{COMPARISON_BASE} -- re-examine check_variant1_scope.py's output "
                  "before deciding this is a scoped, surgical-fix case.")


if __name__ == "__main__":
    main()