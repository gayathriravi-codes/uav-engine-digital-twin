"""
check_variant5_scope.py

v10 follow-up -- is variant 5 (hidden_size=48, dropout=0.15, dropped
['egt','fuel_flow']) uniquely bad on base 008 specifically, or is it a
generally poor variant that happens to get caught here?

check_base008_per_variant.py found variant 5 is a massive outlier on base
008's bucket(100,200) windows (rmse=40.35 vs. next-worst 15.11, bias=-20.37
vs. next-worst -10.92), while being unremarkable on base 005 (rmse=3.05,
middle of the pack). This checks whether that pattern -- fine in general,
broken specifically on base 008 -- holds across the FULL val set (all
buckets, all bases), or whether variant 5 is actually a broadly bad variant
whose damage happens to show up most starkly in the one bucket we've been
looking closely at.

This determines the fix:
  - If variant 5 is uniquely bad on base 008 across buckets -> genuine
    large-capacity-overfits-a-specific-hard-trajectory finding, document it
    (same spirit as the (250,350) known caveat). The ensemble's min-based
    lower bound already partially protects against this; the mean-based
    point estimate does not.
  - If variant 5 is broadly bad everywhere -> treat as an unlucky seed/init
    given hidden_size=48 (the largest in the ensemble) -- consider
    re-seeding just that variant rather than accepting it as "diversity."

Uses the already-trained, already-saved ensemble via load_ensemble() --
does not retrain anything. Val-only. Does NOT touch test.

Run: python models\\rul\\check_variant5_scope.py
"""
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rul import load_all_flights, DEVICE
from train_rul_ensemble import (
    base_trajectory_id,
    build_windowed_dataset_v6,
    load_ensemble,
    SEEDS, DROPOUTS, HIDDEN_SIZES, N_DROPPED_FEATURES,
)
from schema import SENSOR_FIELDS
from sklearn.model_selection import train_test_split

BUCKETS = [(0, 50), (50, 100), (100, 200), (200, 350)]
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
    n_val_bases = sorted(set(val_bases_arr))
    print(f"Val bases present: {n_val_bases}")

    print("\nLoading the already-trained, already-saved ensemble (no retraining) ...")
    models, scaler, dropped_idx_list = load_ensemble()

    # Precompute every variant's predictions on the FULL val set once.
    print("Running all 7 variants individually on the full val set (this may take a moment) ...")
    all_preds = {}  # variant_idx -> np.array of predictions, aligned to X_val
    for v_idx, (model, dropped_idx) in enumerate(zip(models, dropped_idx_list)):
        preds = np.array([
            predict_single_variant(model, scaler, dropped_idx, X_val[i])
            for i in range(len(X_val))
        ])
        all_preds[v_idx] = preds
    print("Done.\n")

    print("=" * 90)
    print(f"VARIANT {VARIANT_OF_INTEREST} vs. ENSEMBLE-MEDIAN-OF-OTHERS RMSE, BY BUCKET x BASE")
    print("=" * 90)
    print("(median-of-others = median RMSE across the other 6 variants, for the same bucket/base slice --")
    print(" a robust 'what's normal here' baseline that isn't skewed by variant 5 itself if it IS the outlier)")
    print()

    header = f"{'bucket':>12} {'base':>6} {'n':>5} {'v5_rmse':>9} {'others_med':>11} {'ratio':>7}"
    print(header)
    print("-" * len(header))

    per_base_v5_rmse = defaultdict(list)   # base -> list of (bucket, v5_rmse, ratio)
    flagged = []

    for lo, hi in BUCKETS:
        bucket_mask = (y_val >= lo) & (y_val < hi)
        for base in n_val_bases:
            idx = np.where(bucket_mask & (val_bases_arr == base))[0]
            if len(idx) == 0:
                continue
            true_vals = y_val[idx]

            variant_rmses = {}
            for v_idx in range(len(models)):
                preds = all_preds[v_idx][idx]
                variant_rmses[v_idx] = np.sqrt(np.mean((preds - true_vals) ** 2))

            v5_rmse = variant_rmses[VARIANT_OF_INTEREST]
            others = [r for v, r in variant_rmses.items() if v != VARIANT_OF_INTEREST]
            others_median = float(np.median(others))
            ratio = v5_rmse / others_median if others_median > 1e-6 else float('inf')

            print(f"{f'({lo},{hi})':>12} {base:>6} {len(idx):>5} {v5_rmse:>9.2f} {others_median:>11.2f} {ratio:>6.2f}x")

            per_base_v5_rmse[base].append((f"({lo},{hi})", v5_rmse, ratio))
            if ratio > 2.5:
                flagged.append((f"({lo},{hi})", base, v5_rmse, others_median, ratio))

    print("\n" + "=" * 90)
    print("SLICES WHERE VARIANT 5 IS >2.5x WORSE THAN THE MEDIAN OF THE OTHER 6 VARIANTS")
    print("=" * 90)
    if flagged:
        for bucket, base, v5_rmse, others_med, ratio in flagged:
            print(f"  bucket={bucket:>10}  base={base}  v5_rmse={v5_rmse:6.2f}  others_median={others_med:6.2f}  ratio={ratio:5.2f}x")
    else:
        print("  (none -- variant 5 is not a >2.5x outlier in any bucket/base slice)")

    flagged_bases = set(b for _, b, _, _, _ in flagged)
    all_bases_touched = set(b for b in n_val_bases)

    print("\n" + "=" * 90)
    print("VERDICT")
    print("=" * 90)
    if len(flagged) == 0:
        print("Variant 5 does not stand out anywhere else -- the base-008/bucket(100,200) result may have")
        print("been a fluke of that specific slice. Re-run check_base008_per_variant.py to confirm it")
        print("still reproduces before concluding anything.")
    elif flagged_bases == {"008"}:
        print("Variant 5's damage is CONCENTRATED ON BASE 008 specifically, across bucket(s):")
        print(f"  {[b for b, base, *_ in flagged if base == '008']}")
        print("-> This supports the 'genuinely hard trajectory for this model family' explanation, NOT a")
        print("   broadly bad variant. Variant 5 (hidden_size=48, the ensemble's largest capacity) likely")
        print("   overfit training data in a way that doesn't generalize to base 008's specific shape.")
        print("   Recommended: document this as a known limitation (same spirit as the (250,350) caveat),")
        print("   and note that the ensemble's MIN-based lower_bound already partially protects against")
        print("   this variant's damage even though the MEAN-based point_estimate does not fully.")
    elif len(flagged_bases) > 1 or len(flagged) > 3:
        print(f"Variant 5's damage spans MULTIPLE bases/buckets: {sorted(flagged_bases)}")
        print("-> This does NOT look like a base-008-specific problem -- variant 5 may be a broadly")
        print("   undertrained or poorly-seeded variant. Recommended: retrain just variant 5 with a")
        print("   different seed and re-run this check, rather than accepting it as ensemble diversity.")
    else:
        print(f"Mixed pattern -- flagged slices: {flagged}")
        print("-> Inspect individually; doesn't cleanly match either the base-008-specific or the")
        print("   broadly-bad-variant explanation.")


if __name__ == "__main__":
    main()