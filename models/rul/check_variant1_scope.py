"""
check_variant1_scope.py

v10 follow-up -- now that re-seeding variant 5 fixed base 008's original
outlier (RMSE 40.35 -> 1.16), check_ensemble_bucket100_200_breakdown.py and
check_base008_per_variant.py show variant 1 (seed=1, drops ['cht']) as the
new worst variant on base 008/bucket(100,200): RMSE 15.11, bias -10.92 --
worse than any variant on either base, including base 008's other variants.

This script mirrors check_variant5_scope.py's original question, adapted
to variant 1: is variant 1's badness SCOPED to base 008/bucket(100,200)
specifically (same shape as variant 5's original problem -- a re-seed
candidate), or does it show up broadly across other buckets/bases too
(suggesting a generally weak/undertrained variant instead)?

Method: for every (bucket, base) slice with val windows, run variant 1
individually (bypassing predict_rul_ensemble's averaging, same as
check_base008_per_variant.py) and compare its RMSE to the MEDIAN RMSE of
the other 6 variants on that same slice. Flag ratio >= 2.5x.

Built directly on check_base008_per_variant.py's verified split-reproduction
logic (same imports, same train_test_split(unique_bases, ...) sequence) --
NOT reimplemented independently, per standing project discipline.

Uses the already-trained, already-saved ensemble via load_ensemble() --
does not retrain anything. Val-only. Does NOT touch test.

Run: python models\\rul\\check_variant1_scope.py
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
    SEEDS, DROPOUTS, HIDDEN_SIZES,
)
from schema import SENSOR_FIELDS
from calibrate_rul import BUCKET_EDGES
from sklearn.model_selection import train_test_split

TARGET_VARIANT = 1
RATIO_FLAG_THRESHOLD = 2.5


def predict_single_variant(model, scaler, dropped_idx, window):
    """Same as check_base008_per_variant.py: runs ONE variant model on ONE
    window, bypassing predict_rul_ensemble's averaging."""
    scaled = scaler.transform(window.reshape(-1, window.shape[-1])).reshape(window.shape)
    window_variant = scaled.copy()
    if len(dropped_idx) > 0:
        window_variant[:, dropped_idx] = 0.0
    model.eval()
    with torch.no_grad():
        x = torch.tensor(window_variant, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        return model(x).item()


def rmse(true_vals, preds):
    return float(np.sqrt(np.mean((preds - true_vals) ** 2)))


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
    val_base_ids = np.array([base_trajectory_id(f) for f in val_flight_ids])

    print("\nLoading the already-trained, already-saved ensemble (no retraining) ...")
    models, scaler, dropped_idx_list = load_ensemble()
    n_variants = len(models)
    other_variants = [v for v in range(n_variants) if v != TARGET_VARIANT]

    print("\n" + "=" * 88)
    print(f"VARIANT {TARGET_VARIANT} SCOPE CHECK -- all buckets x all val bases")
    print("=" * 88)
    print(f"{'bucket':>12} {'base':>6} {'n':>5} {'v'+str(TARGET_VARIANT)+'_rmse':>10} "
          f"{'others_med_rmse':>16} {'ratio':>7}  flag")

    flagged = []
    for lo, hi in BUCKET_EDGES:
        bucket_mask = (y_val >= lo) & (y_val < hi)
        for base in val_bases_set:
            idx = np.array([
                i for i in np.where(bucket_mask)[0]
                if val_base_ids[i] == base
            ])
            n = len(idx)
            if n == 0:
                continue
            true_vals = y_val[idx]

            target_preds = np.array([
                predict_single_variant(models[TARGET_VARIANT], scaler,
                                        dropped_idx_list[TARGET_VARIANT], X_val[i])
                for i in idx
            ])
            v_rmse = rmse(true_vals, target_preds)

            other_rmses = []
            for v in other_variants:
                other_preds = np.array([
                    predict_single_variant(models[v], scaler, dropped_idx_list[v], X_val[i])
                    for i in idx
                ])
                other_rmses.append(rmse(true_vals, other_preds))
            med_other = float(np.median(other_rmses))
            ratio = v_rmse / med_other if med_other > 0 else float("inf")

            flag = "  <-- FLAG" if ratio >= RATIO_FLAG_THRESHOLD else ""
            if flag:
                flagged.append((f"({lo},{hi})", base, ratio))

            print(f"{f'({lo},{hi})':>12} {base:>6} {n:>5} {v_rmse:>10.2f} "
                  f"{med_other:>16.2f} {ratio:>6.2f}x{flag}")

    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    if len(flagged) == 0:
        print(f"Variant {TARGET_VARIANT} is not >={RATIO_FLAG_THRESHOLD}x worse than the "
              "median of the other variants in ANY slice.")
        print("-> Does not show a scoped interaction. Re-seeding may still help overall, "
              "but this is not the same clean shape as variant 5's case -- check whether "
              "variant 1 is just generally a bit weak rather than having one bad "
              "feature-bagging interaction.")
    elif len(flagged) == 1:
        slice_name, base, ratio = flagged[0]
        print(f"Variant {TARGET_VARIANT} is scoped: only flagged in "
              f"bucket{slice_name} x base {base} (ratio {ratio:.2f}x).")
        print("-> Same shape as variant 5's original problem. Re-seeding variant 1 is "
              "a reasonable, low-risk next step. Proceed the same way: re-seed -> "
              "re-run check_base008_per_variant.py -> re-run "
              "check_ensemble_bucket100_200_breakdown.py -> refit calibration -> only "
              "then re-touch test.")
    else:
        print(f"Variant {TARGET_VARIANT} is flagged in {len(flagged)} slices:")
        for slice_name, base, ratio in flagged:
            print(f"    bucket{slice_name} x base {base}: {ratio:.2f}x")
        print("-> Broader problem than a single scoped interaction. Re-seeding may still "
              "help but check whether variant 1 is generally undertrained (final train "
              "loss vs. the other 6) before assuming a seed swap alone fixes it -- and "
              "check whether fixing this slice reintroduces a regression in one of the "
              "other flagged slices.")


if __name__ == "__main__":
    main()