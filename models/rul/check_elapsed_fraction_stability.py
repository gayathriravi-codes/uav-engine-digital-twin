"""
check_elapsed_fraction_stability.py

Bug 2 track — verification before trusting train_rul_elapsed_fraction_experiment.py's
result (bucket(100,200) max overshoot +52.39 -> +0.12). That magnitude is large
enough to warrant the same suspicion the reweighting experiment's fake "fix"
deserved -- this checks:

  1. Per-flight/per-base breakdown: is the improvement general across bucket
     (100,200)'s val windows, or concentrated on 1-2 flights/bases (the exact
     failure mode that made the reweighting experiment look better than it was)?
  2. Multi-seed stability: does bucket(100,200) max overshoot stay low across
     several seeds, or was seed=42 a lucky draw?

Reuses the identical dataset-building / split logic from the experiment script
(imported, not reimplemented) so results are directly comparable.

Val-only. Does NOT touch test.
Run: python models\\rul\\check_elapsed_fraction_stability.py
"""
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rul import load_all_flights, RULDataset, DEVICE
from train_rul_ensemble import base_trajectory_id, bootstrap_by_flight
from train_rul_elapsed_fraction_experiment import (
    build_dataset_with_elapsed_fraction, build_dataset_baseline,
    RULRegressorExperimental, HIDDEN_SIZE, DROPOUT, EPOCHS, BATCH_SIZE, LR,
)
import torch.nn as nn
from torch.utils.data import DataLoader

BUCKET_LO, BUCKET_HI = 100, 200
SEEDS = [42, 7, 123, 2024]


def train_variant(X_train, y_train, train_flight_ids, n_features, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train_boot, y_train_boot = bootstrap_by_flight(X_train, y_train, train_flight_ids, seed)
    train_loader = DataLoader(RULDataset(X_train_boot, y_train_boot), batch_size=BATCH_SIZE, shuffle=True)

    model = RULRegressorExperimental(n_features=n_features, hidden_size=HIDDEN_SIZE, dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
    return model


def predict_all(model, X_eval):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(X_eval)):
            x = torch.tensor(X_eval[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            preds.append(model(x).item())
    return np.array(preds)


def main():
    print("Loading flights ...")
    flights = load_all_flights()

    print("Building datasets (baseline + elapsed_fraction, identical window boundaries) ...")
    X_base, y_base, flight_ids_base = build_dataset_baseline(flights)
    X_ef, y_ef, flight_ids_ef = build_dataset_with_elapsed_fraction(flights)
    assert len(X_base) == len(X_ef) and np.allclose(y_base, y_ef)

    unique_flights = np.unique(flight_ids_base)
    base_ids = np.array([base_trajectory_id(f) for f in unique_flights])
    unique_bases = np.unique(base_ids)

    train_bases, temp_bases = train_test_split(unique_bases, test_size=0.3, random_state=42)
    val_bases, test_bases = train_test_split(temp_bases, test_size=0.5, random_state=42)
    train_bases_set, val_bases_set = set(train_bases), set(val_bases)

    train_flights = np.array([f for f in unique_flights if base_trajectory_id(f) in train_bases_set])
    val_flights = np.array([f for f in unique_flights if base_trajectory_id(f) in val_bases_set])

    train_mask = np.isin(flight_ids_base, train_flights)
    val_mask = np.isin(flight_ids_base, val_flights)
    train_flight_ids = flight_ids_base[train_mask]
    val_flight_ids_arr = flight_ids_base[val_mask]

    X_train, y_train = X_ef[train_mask], y_ef[train_mask]
    X_val, y_val = X_ef[val_mask], y_ef[val_mask]
    n_features = X_train.shape[-1]

    bucket_mask_val = (y_val >= BUCKET_LO) & (y_val < BUCKET_HI)
    print(f"\nVal windows in bucket ({BUCKET_LO},{BUCKET_HI}): {bucket_mask_val.sum()}\n")

    print("=" * 78)
    print("MULTI-SEED STABILITY (elapsed_fraction variant, same split, different seeds)")
    print("=" * 78)

    seed_results = []
    for seed in SEEDS:
        scaler = StandardScaler()
        scaler.fit(X_train.reshape(-1, n_features))
        X_train_scaled = scaler.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
        X_val_scaled = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)

        model = train_variant(X_train_scaled, y_train, train_flight_ids, n_features, seed)
        preds = predict_all(model, X_val_scaled)

        bucket_preds = preds[bucket_mask_val]
        bucket_true = y_val[bucket_mask_val]
        bucket_rmse = np.sqrt(np.mean((bucket_preds - bucket_true) ** 2))
        bucket_max_overshoot = np.max(bucket_preds - bucket_true)
        bucket_bias = np.mean(bucket_preds - bucket_true)

        seed_results.append({
            "seed": seed, "rmse": bucket_rmse, "max_overshoot": bucket_max_overshoot,
            "bias": bucket_bias, "preds": preds,
        })
        print(f"  seed={seed:5d}  bucket_rmse={bucket_rmse:6.2f}  "
              f"max_overshoot={bucket_max_overshoot:+7.2f}  bias={bucket_bias:+7.2f}")

    rmses = [r["rmse"] for r in seed_results]
    overshoots = [r["max_overshoot"] for r in seed_results]
    print(f"\n  bucket_rmse across seeds:       min={min(rmses):.2f}  max={max(rmses):.2f}  "
          f"spread={max(rmses)-min(rmses):.2f}")
    print(f"  max_overshoot across seeds:     min={min(overshoots):+.2f}  max={max(overshoots):+.2f}  "
          f"spread={max(overshoots)-min(overshoots):.2f}")

    print("\n" + "=" * 78)
    print("PER-FLIGHT / PER-BASE BREAKDOWN (seed=42, matching the original experiment run)")
    print("=" * 78)

    preds_42 = seed_results[0]["preds"]
    bucket_idx = np.where(bucket_mask_val)[0]

    print("\n-- RMSE and max overshoot by flight, within bucket (100,200) --")
    by_flight = defaultdict(lambda: {"true": [], "pred": []})
    for i in bucket_idx:
        fid = val_flight_ids_arr[i]
        by_flight[fid]["true"].append(y_val[i])
        by_flight[fid]["pred"].append(preds_42[i])

    for fid, d in sorted(by_flight.items()):
        true_arr = np.array(d["true"])
        pred_arr = np.array(d["pred"])
        rmse = np.sqrt(np.mean((pred_arr - true_arr) ** 2))
        max_over = np.max(pred_arr - true_arr)
        print(f"  {fid:30s}: n={len(true_arr):3d}  rmse={rmse:6.2f}  max_overshoot={max_over:+7.2f}")

    print("\n-- Same, grouped by base trajectory --")
    by_base = defaultdict(lambda: {"true": [], "pred": []})
    for i in bucket_idx:
        base = base_trajectory_id(val_flight_ids_arr[i])
        by_base[base]["true"].append(y_val[i])
        by_base[base]["pred"].append(preds_42[i])

    for base, d in sorted(by_base.items()):
        true_arr = np.array(d["true"])
        pred_arr = np.array(d["pred"])
        rmse = np.sqrt(np.mean((pred_arr - true_arr) ** 2))
        max_over = np.max(pred_arr - true_arr)
        print(f"  base {base}: n={len(true_arr):3d}  rmse={rmse:6.2f}  max_overshoot={max_over:+7.2f}")

    # Specifically re-check base 005's known val anomaly (flat point_est ~210-214
    # regardless of true_rul) -- did elapsed_fraction actually fix THIS, or did
    # it just improve the aggregate while this specific known-bad pattern persists?
    print("\n-- Base 005's previously-known anomalous windows (true_rul~165, was flat ~210-214) --")
    base005_idx = [i for i in bucket_idx if base_trajectory_id(val_flight_ids_arr[i]) == "005"
                   and abs(y_val[i] - 165.0) < 1e-6]
    for i in base005_idx[:6]:
        print(f"  {val_flight_ids_arr[i]:30s}  true={y_val[i]:.1f}  pred={preds_42[i]:.1f}  "
              f"error={preds_42[i]-y_val[i]:+.1f}")

    print("\nSTOP-OR-CONTINUE: if RMSE/max_overshoot swing widely across seeds, or the")
    print("improvement is concentrated on 1-2 flights/bases while others stay bad, treat")
    print("this the same as the reweighting experiment -- looks good in aggregate, isn't")
    print("a general fix. Only wire elapsed_fraction into the full ensemble if BOTH checks")
    print("come back clean.")


if __name__ == "__main__":
    main()