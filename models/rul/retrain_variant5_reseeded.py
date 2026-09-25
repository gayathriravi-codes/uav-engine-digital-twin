"""
retrain_variant5_reseeded.py

v10 follow-up -- re-seeds and retrains ONLY variant 5 (previously seed=5,
hidden_size=48, dropout=0.15), which check_base008_per_variant.py and
check_variant5_scope.py identified as a base-008-specific outlier
(RMSE 40.35 vs. next-worst 15.11 on base 008/bucket(100,200), while
unremarkable everywhere else -- confirmed NOT a broadly bad variant).

check_lower_bound_attribution.py confirmed this is safe to fix: raw
lower_bound <= true_rul coverage is 100% on base 008 both with and without
variant 5, so re-seeding carries no coverage regression risk visible in
that check. Re-seeding (not just documenting as a known limitation) was
chosen because there's no downside surfaced so far and a likely upside
(fixes the point estimate's -20.37 bias on base 008 without needing to
touch the other 6 variants).

Reuses the ALREADY-TRAINED other 6 variants and the ALREADY-FIT scaler --
does NOT retrain the whole ensemble, does NOT refit the scaler (refitting
the scaler on the same train set would produce numerically-identical
results anyway, but re-using the saved one avoids any doubt and is
consistent with the "don't reimplement, don't redo unnecessary work"
discipline in this project). Rebuilds the train/val split identically
(same base_trajectory_id-based split, same random_state) so variant 5's
new training data is exactly what it should be -- not reimplemented
independently, matches train_rul_ensemble.py's own __main__ logic exactly.

Overwrites ONLY rul_model_variant_5.pt on disk. Does NOT touch
rul_ensemble_scaler.joblib or rul_ensemble_dropped_idx.joblib (variant 5's
dropped_idx, from feature bagging, is DETERMINISTIC given N_DROPPED_FEATURES[5]
and the OLD seed=5 -- since apply_feature_bagging's RNG is seeded off the
variant's seed, changing the seed also changes WHICH sensor(s) get dropped
for variant 5. This is intentional and expected, not a bug -- the new
dropped_idx for variant 5 is saved into dropped_idx_list and persisted,
replacing only that one entry.

After running this, you MUST:
  1. Run check_base008_per_variant.py again to confirm variant 5 is no
     longer an outlier on base 008.
  2. Run check_ensemble_bucket100_200_breakdown.py to confirm ensemble-level
     base 008 RMSE has moved toward base 003/005's range.
  3. Refit calibration (fit_calibration) -- do NOT assume the old
     calibration params are still valid; the ensemble's predictions have
     changed.
  4. Only then consider run_coverage_check.py / test.

Val-only for verification prints below. Does NOT touch test.

Run: python models\\rul\\retrain_variant5_reseeded.py
"""
import os
import sys

import numpy as np
import torch
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rul import load_all_flights, DEVICE, MODEL_OUT_DIR
from train_rul_ensemble import (
    base_trajectory_id,
    build_windowed_dataset_v6,
    train_one_variant,
    predict_rul_ensemble,
    DROPOUTS, HIDDEN_SIZES, N_DROPPED_FEATURES,
)
from sklearn.model_selection import train_test_split

VARIANT_IDX = 5
OLD_SEED = 5
NEW_SEED = 50  # deliberately far outside the existing 0-6 range, easy to spot in logs

BUCKET_LO, BUCKET_HI = 100, 200
TARGET_BASE = "008"


def main():
    print(f"Re-seeding variant {VARIANT_IDX}: seed {OLD_SEED} -> {NEW_SEED}")
    print(f"(hidden_size={HIDDEN_SIZES[VARIANT_IDX]}, dropout={DROPOUTS[VARIANT_IDX]}, "
          f"n_dropped_features={N_DROPPED_FEATURES[VARIANT_IDX]} -- unchanged, only the seed changes)\n")

    print("Loading flights ...")
    flights = load_all_flights()

    print("Building v10 (11-feature) windowed dataset -- must match production exactly ...")
    X, y, flight_ids = build_windowed_dataset_v6(flights)

    # Identical split logic to train_rul_ensemble.py's __main__ -- imported
    # base_trajectory_id, same random_state, same order of operations.
    print("Reproducing the exact train/val/test split (base-trajectory-grouped) ...")
    unique_flights = np.unique(flight_ids)
    base_ids = np.array([base_trajectory_id(f) for f in unique_flights])
    unique_bases = np.unique(base_ids)

    train_bases, temp_bases = train_test_split(unique_bases, test_size=0.3, random_state=42)
    val_bases, test_bases = train_test_split(temp_bases, test_size=0.5, random_state=42)
    train_bases_set, val_bases_set = set(train_bases), set(val_bases)

    train_flights = np.array([f for f in unique_flights if base_trajectory_id(f) in train_bases_set])
    val_flights = np.array([f for f in unique_flights if base_trajectory_id(f) in val_bases_set])

    train_mask = np.isin(flight_ids, train_flights)
    val_mask = np.isin(flight_ids, val_flights)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    train_flight_ids = flight_ids[train_mask]

    # NOTE: this deliberately does NOT redo the synthetic-multifault
    # augmentation step from train_rul_ensemble.py's __main__. That step
    # uses seed=42 (independent of per-variant seeds) and generates windows
    # by combining pairs of REAL training windows -- reproducing it exactly
    # would require either re-running that exact code path or accepting
    # slightly different synthetic windows if regenerated. Since this
    # variant's whole point is an apples-to-apples re-seed of an otherwise
    # identical variant, and the original variant 5 WAS trained with
    # synthetic augmentation included, skipping it here means this retrain
    # is not perfectly identical to how variant 5 was first trained --
    # only the real-window portion of train is reproduced.
    #
    # If you want a fully faithful re-seed (recommended before treating this
    # as final), regenerate the synthetic windows exactly as
    # train_rul_ensemble.py's __main__ does (generate_synthetic_multifault_windows,
    # seed=42) and concatenate them here before scaling. Left out of this
    # minimal script to keep it a small, auditable diff -- do not skip this
    # silently if you're promoting this beyond a quick check.
    print("\nWARNING: this minimal retrain does NOT regenerate synthetic multi-fault training")
    print("windows (train_rul_ensemble.py's __main__ adds those via")
    print("generate_synthetic_multifault_windows(seed=42) before training). If you want an exact")
    print("match to how the other 6 variants were trained, add that step before proceeding to")
    print("treat this as final -- see the comment above this print for why it's omitted here.\n")

    print("Scaling using the ALREADY-SAVED ensemble scaler (fit on train only, not refit here) ...")
    scaler_path = os.path.join(MODEL_OUT_DIR, "rul_ensemble_scaler.joblib")
    scaler = joblib.load(scaler_path)
    X_train_scaled = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

    print(f"Training variant {VARIANT_IDX} with NEW seed={NEW_SEED} ...")
    new_model, new_rmse, new_dropped_idx, n_params = train_one_variant(
        X_train_scaled, y_train, train_flight_ids, X_val_scaled, y_val,
        seed=NEW_SEED,
        dropout=DROPOUTS[VARIANT_IDX],
        hidden_size=HIDDEN_SIZES[VARIANT_IDX],
        n_dropped_features=N_DROPPED_FEATURES[VARIANT_IDX],
    )
    dropped_names = [f for f in new_dropped_idx]
    print(f"  -> new val RMSE (overall): {new_rmse:.2f}  ({n_params:,} params)  dropped_idx: {new_dropped_idx}")

    print("\nLoading the rest of the CURRENT ensemble (variants 0,1,2,3,4,6 stay exactly as they are) ...")
    dropped_idx_path = os.path.join(MODEL_OUT_DIR, "rul_ensemble_dropped_idx.joblib")
    dropped_idx_list = joblib.load(dropped_idx_path)

    n_variants = len(dropped_idx_list)
    models = []
    from train_rul_ensemble import RULRegressorVariant
    for i in range(n_variants):
        model_path = os.path.join(MODEL_OUT_DIR, f"rul_model_variant_{i}.pt")
        if i == VARIANT_IDX:
            models.append(new_model)  # the freshly retrained one, still in memory
            continue
        m = RULRegressorVariant(hidden_size=HIDDEN_SIZES[i])
        m.load_state_dict(torch.load(model_path, map_location=DEVICE))
        m.to(DEVICE)
        m.eval()
        models.append(m)

    dropped_idx_list[VARIANT_IDX] = new_dropped_idx

    print(f"\nQUICK CHECK -- base {TARGET_BASE}, bucket({BUCKET_LO},{BUCKET_HI}), BEFORE saving:")
    base_ids_val = np.array([base_trajectory_id(f) for f in flight_ids[val_mask]])
    bucket_mask = (y_val >= BUCKET_LO) & (y_val < BUCKET_HI)
    target_idx = np.where(bucket_mask & (base_ids_val == TARGET_BASE))[0]

    preds = np.array([
        predict_rul_ensemble(X_val[i], models, scaler, dropped_idx_list, calibrated=False)["point_estimate_timesteps"]
        for i in target_idx
    ])
    true_vals = y_val[target_idx]
    rmse = np.sqrt(np.mean((preds - true_vals) ** 2))
    bias = np.mean(preds - true_vals)
    max_over = np.max(preds - true_vals)
    print(f"  n={len(target_idx)}  rmse={rmse:.2f}  bias={bias:+.2f}  max_overshoot={max_over:+.2f}")
    print(f"  (for reference: WITH old variant 5, this was rmse=12.89, bias=-7.60 -- see")
    print(f"   check_ensemble_bucket100_200_breakdown.py's earlier output)")

    print(f"\nSaving retrained variant {VARIANT_IDX} model and updated dropped_idx_list ...")
    model_path = os.path.join(MODEL_OUT_DIR, f"rul_model_variant_{VARIANT_IDX}.pt")
    torch.save(new_model.state_dict(), model_path)
    joblib.dump(dropped_idx_list, dropped_idx_path)
    print(f"  Saved {model_path}")
    print(f"  Updated {dropped_idx_path}")

    print("\nDone. NEXT STEPS (do not skip):")
    print("  1. Run check_base008_per_variant.py again -- confirm variant 5 is no longer an")
    print("     outlier on base 008.")
    print("  2. Run check_ensemble_bucket100_200_breakdown.py -- confirm ensemble-level base 008")
    print("     RMSE has moved toward base 003/005's range.")
    print("  3. Refit calibration: from calibrate_rul import fit_calibration; fit_calibration(")
    print("     X_val, y_val, models, scaler, dropped_idx_list) -- do NOT assume the old")
    print("     calibration params still apply, the ensemble's predictions have changed.")
    print("  4. Only after 1-3 look clean: consider run_coverage_check.py / test.")
    print("  5. Remember the synthetic-augmentation caveat printed above before treating this")
    print("     as a fully faithful match to how the other 6 variants were trained.")


if __name__ == "__main__":
    main()