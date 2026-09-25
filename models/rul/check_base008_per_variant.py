"""
check_base008_per_variant.py

v10 follow-up -- isolating WHY base trajectory 008 is ~4-7x worse than
bases 003/005 in the retrained 7-variant ensemble's bucket(100,200)
predictions (ensemble RMSE 12.89 vs. 5.04/1.86; single-variant experiment
had 008 already weakest but only 3.08-3.83, not 12.7-13.7).

Two competing explanations to distinguish:
  (a) One or two SPECIFIC variants are wildly off on base 008 (likely a
      feature-bagging combination -- e.g. dropping a sensor that happens
      to be load-bearing for base 008's fault signature specifically)
      while the rest are fine. This would be a fixable/investigatable
      regression, analogous in shape to why v7's OOD fix silently didn't
      fire, or why the reweighting experiment's fix was fake.
  (b) All 7 variants are mediocre-but-consistent on base 008 -- suggesting
      this is a genuinely harder trajectory shape that needs more
      capacity/signal generally, not a specific bug in one variant.

Does this by calling each model in the loaded ensemble INDIVIDUALLY
(bypassing predict_rul_ensemble's averaging) on base 008's bucket(100,200)
val windows, plus the same for base 005 as a healthy-comparison baseline.

Uses the already-trained, already-saved ensemble via load_ensemble() --
does not retrain anything. Val-only. Does NOT touch test.

Run: python models\\rul\\check_base008_per_variant.py
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
    SEEDS, DROPOUTS, HIDDEN_SIZES, N_DROPPED_FEATURES,
)
from schema import SENSOR_FIELDS
from sklearn.model_selection import train_test_split

BUCKET_LO, BUCKET_HI = 100, 200
TARGET_BASE = "008"
COMPARISON_BASE = "005"  # healthiest base in the ensemble breakdown, as a baseline


def predict_single_variant(model, scaler, dropped_idx, window):
    """Runs ONE variant model on ONE window, bypassing predict_rul_ensemble's
    averaging across variants -- so each variant's individual behavior on
    base 008 is visible, not smoothed into the ensemble mean."""
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

    # Identical split to train_rul_ensemble.py / check_ensemble_bucket100_200_breakdown.py
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

    print("\nLoading the already-trained, already-saved ensemble (no retraining) ...")
    models, scaler, dropped_idx_list = load_ensemble()

    for target_base, label in [(TARGET_BASE, "TARGET (known-bad)"), (COMPARISON_BASE, "COMPARISON (healthy)")]:
        base_idx = np.array([
            i for i in np.where(bucket_mask)[0]
            if base_trajectory_id(val_flight_ids[i]) == target_base
        ])
        true_vals = y_val[base_idx]

        print("\n" + "=" * 78)
        print(f"BASE {target_base} -- {label} -- n={len(base_idx)} windows in bucket({BUCKET_LO},{BUCKET_HI})")
        print("=" * 78)
        print(f"{'variant':>8} {'seed':>5} {'hidden':>7} {'dropout':>8} {'dropped_sensors':>25} "
              f"{'rmse':>7} {'max_over':>9} {'bias':>8}")

        variant_rmses = []
        for v_idx, (model, dropped_idx) in enumerate(zip(models, dropped_idx_list)):
            preds = np.array([
                predict_single_variant(model, scaler, dropped_idx, X_val[i])
                for i in base_idx
            ])
            rmse = np.sqrt(np.mean((preds - true_vals) ** 2))
            max_over = np.max(preds - true_vals)
            bias = np.mean(preds - true_vals)
            variant_rmses.append(rmse)
            dropped_names = [SENSOR_FIELDS[j] for j in dropped_idx]
            print(f"{v_idx:>8} {SEEDS[v_idx]:>5} {HIDDEN_SIZES[v_idx]:>7} {DROPOUTS[v_idx]:>8} "
                  f"{str(dropped_names):>25} {rmse:>7.2f} {max_over:>+9.2f} {bias:>+8.2f}")

        spread = max(variant_rmses) - min(variant_rmses)
        print(f"\n  Per-variant RMSE for base {target_base}: min={min(variant_rmses):.2f}  "
              f"max={max(variant_rmses):.2f}  spread={spread:.2f}")
        if spread > 10:
            print(f"  -> WIDE spread: one or more SPECIFIC variants are much worse than the rest on this base.")
            print(f"     Check which sensor(s) those variants dropped -- likely a feature-bagging combination")
            print(f"     that happens to remove signal this base's fault type depends on.")
        else:
            print(f"  -> TIGHT spread: all variants perform similarly on this base -- consistent with a")
            print(f"     genuinely harder trajectory shape rather than one bad variant.")

    print("\n" + "=" * 78)
    print("INTERPRETATION GUIDE")
    print("=" * 78)
    print("Compare base 008's per-variant spread to base 005's directly above.")
    print("If base 008 shows a WIDE per-variant spread (a few variants much worse than the")
    print("rest) while base 005 is TIGHT across all variants -- that points to (a): specific")
    print("variants (likely ones with a particular feature dropped) failing on base 008's")
    print("fault signature specifically. Cross-reference the dropped_sensors column: if the")
    print("worst variants on base 008 share a dropped sensor that the best variants keep,")
    print("that sensor is likely load-bearing for whatever fault type(s) base 008 carries.")
    print("If BOTH bases show tight per-variant spread, but base 008's overall level is just")
    print("uniformly higher than base 005's across every variant -- that points to (b): base")
    print("008 is a genuinely harder trajectory for this model family in general, not a bug")
    print("in one variant. That's a real finding worth documenting (same spirit as the")
    print("(250,350) known-caveat bucket), not something today's check will \"fix\" -- it")
    print("would need either more capacity, more training signal for that fault-type/shape,")
    print("or an explicit documented limitation, same three options bucket(100,200) itself")
    print("faced before elapsed_fraction was tried.")


if __name__ == "__main__":
    main()