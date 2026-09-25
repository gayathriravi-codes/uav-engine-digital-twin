"""
check_multiseed_stability.py

Multi-seed stability check for the reweighted-loss experiment. A single
seed (42) showed the bucket-wide "bias reduced" result was actually an
averaging artifact -- some flights (cooling_degradation_*) genuinely
improved, others (oil_issue_008, sensor_drift_009) got much worse, and
one (sensor_drift_011) just flipped error sign. Before concluding
anything about reweighting, this checks whether that per-flight pattern
holds across seeds or was this-seed noise.

Also tests two variants per seed:
  - reweighted, LOW_RUL_WEIGHT=3.0 (as before)
  - reweighted, LOW_RUL_WEIGHT=1.5 (gentler)
both with predictions clamped to >=0 at eval time (does not change what
the model learns, only reports what a clamped-output production pipeline
would actually show -- clamping is already effectively applied downstream
via the calibration lower-bound, so this makes the comparison closer to
what coverage checks would see).

Val-only. Does NOT touch test. Does NOT save models. This trains
baseline + 2 weighted variants x N_SEEDS models total -- slower than
prior scripts, expect several minutes.

Run: python models\\rul\\check_multiseed_stability.py
"""
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rul import load_all_flights, RULDataset, DEVICE
from train_rul_ensemble import (
    build_windowed_dataset_v5,
    RULRegressorVariant,
    bootstrap_by_flight,
)

SEEDS = [42, 43, 44]
HIDDEN_SIZE = 32
DROPOUT = 0.1
EPOCHS = 70
BATCH_SIZE = 32
LR = 1e-3
LOW_RUL_THRESHOLD = 50.0
WEIGHTS_TO_TEST = [1.5, 3.0]


def weighted_mse_loss(preds, targets, weight_mask, low_weight):
    weights = torch.where(weight_mask, torch.full_like(targets, low_weight), torch.ones_like(targets))
    return (weights * (preds - targets) ** 2).mean()


def train_variant(X_train, y_train, train_flight_ids, seed, weight):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_train_boot, y_train_boot = bootstrap_by_flight(X_train, y_train, train_flight_ids, seed)
    train_loader = DataLoader(RULDataset(X_train_boot, y_train_boot), batch_size=BATCH_SIZE, shuffle=True)
    model = RULRegressorVariant(hidden_size=HIDDEN_SIZE, dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            if weight is None:
                loss = nn.functional.mse_loss(preds, yb)
            else:
                mask = yb < LOW_RUL_THRESHOLD
                loss = weighted_mse_loss(preds, yb, mask, weight)
            loss.backward()
            optimizer.step()
    return model


def predict_all(model, X_eval, clamp=True):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(X_eval)):
            x = torch.tensor(X_eval[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            preds.append(model(x).item())
    preds = np.array(preds)
    if clamp:
        preds = np.clip(preds, 0.0, None)
    return preds


def main():
    print("Loading flights and building windowed dataset ...")
    flights = load_all_flights()
    X, y, flight_ids = build_windowed_dataset_v5(flights)

    unique_flights = np.unique(flight_ids)
    train_flights, temp_flights = train_test_split(unique_flights, test_size=0.3, random_state=42)
    val_flights, test_flights = train_test_split(temp_flights, test_size=0.5, random_state=42)

    train_mask = np.isin(flight_ids, train_flights)
    val_mask = np.isin(flight_ids, val_flights)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    val_flight_ids = flight_ids[val_mask]
    train_flight_ids_arr = flight_ids[train_mask]

    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    X_train_scaled = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

    low_mask = y_val < LOW_RUL_THRESHOLD
    low_idx = np.where(low_mask)[0]
    by_flight = defaultdict(list)
    for i in low_idx:
        by_flight[val_flight_ids[i]].append(i)

    # variant label -> seed -> per-flight results, plus bucket-wide + negative counts
    variants = ["baseline"] + [f"weighted_{w}x" for w in WEIGHTS_TO_TEST]
    flight_results = {v: defaultdict(list) for v in variants}  # flight -> list of (rmse,bias) across seeds
    bucket_results = {v: [] for v in variants}  # list of (rmse,bias) across seeds
    neg_counts = {v: [] for v in variants}  # list of neg-pred counts (bucket<50) across seeds

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        print("  training baseline ...")
        model_base = train_variant(X_train_scaled, y_train, train_flight_ids_arr, seed, weight=None)
        preds_by_variant = {"baseline": predict_all(model_base, X_val_scaled)}
        for w in WEIGHTS_TO_TEST:
            print(f"  training weighted {w}x ...")
            model_w = train_variant(X_train_scaled, y_train, train_flight_ids_arr, seed, weight=w)
            preds_by_variant[f"weighted_{w}x"] = predict_all(model_w, X_val_scaled)

        for v, preds in preds_by_variant.items():
            true_v = y_val[low_mask]
            p = preds[low_mask]
            rmse = np.sqrt(np.mean((p - true_v) ** 2))
            bias = np.mean(p - true_v)
            bucket_results[v].append((rmse, bias))
            neg_counts[v].append(int(np.sum(preds[low_mask] <= 0.0)))  # clamp means 0 = was negative
            for fid, idxs in by_flight.items():
                idxs = np.array(idxs)
                ft = y_val[idxs]
                fp = preds[idxs]
                frmse = np.sqrt(np.mean((fp - ft) ** 2))
                fbias = np.mean(fp - ft)
                flight_results[v][fid].append((frmse, fbias))

    print("\n\n=== Bucket<50 results across seeds (mean +/- std) ===")
    print(f"{'variant':>14} {'RMSE mean':>10} {'RMSE std':>9} {'bias mean':>10} {'bias std':>9} {'clamped-to-0 (mean count)':>26}")
    for v in variants:
        rmses = [r for r, b in bucket_results[v]]
        biases = [b for r, b in bucket_results[v]]
        print(f"{v:>14} {np.mean(rmses):>10.2f} {np.std(rmses):>9.2f} "
              f"{np.mean(biases):>+10.2f} {np.std(biases):>9.2f} {np.mean(neg_counts[v]):>26.1f}")

    print("\n=== Per-flight results across seeds (mean +/- std RMSE, mean bias) ===")
    header = f"{'flight':>22}"
    for v in variants:
        header += f" {v+'_RMSE':>16} {v+'_bias':>12}"
    print(header)
    for fid in sorted(flight_results["baseline"].keys()):
        row = f"{fid:>22}"
        for v in variants:
            vals = flight_results[v][fid]
            rmses = [r for r, b in vals]
            biases = [b for r, b in vals]
            row += f" {np.mean(rmses):>7.2f}+/-{np.std(rmses):<5.2f} {np.mean(biases):>+7.2f}    "
        print(row)

    print("\nHow to read this: for each flight, compare RMSE mean+/-std across variants.")
    print("A flight is a STABLE improvement if weighted RMSE is lower than baseline AND")
    print("the std doesn't overlap baseline's range. If std is large relative to the")
    print("mean difference, that flight's earlier single-seed result was likely noise.")
    print("Same logic for bucket-wide: only trust the bucket-wide mean if per-flight")
    print("results are individually consistent, not just canceling out again.")


if __name__ == "__main__":
    main()