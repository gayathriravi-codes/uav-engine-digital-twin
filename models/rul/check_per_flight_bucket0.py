"""
check_per_flight_bucket0.py

Per-flight breakdown of the baseline vs reweighted models on bucket<50
val windows -- checks whether the "bias reduced" result from
train_rul_reweighted_experiment.py is a genuine improvement spread across
flights, or a collapse on one/few flights (e.g. idx 705-714) canceling
out against still-broken flights (e.g. idx 760-769) in the bucket-wide mean.

Also checks: how many windows (in bucket<50, and overall on val) get a
NEGATIVE prediction from each model -- an unclamped model producing
negative RUL for low-true-RUL inputs is a red flag about what 3x
reweighting is doing near zero, independent of this specific bug.

Val-only. Does NOT touch test. Does NOT save models (retrains both
variants fresh, matching train_rul_reweighted_experiment.py exactly,
so the flight_id association can be tracked per-window).

Run: python models\\rul\\check_per_flight_bucket0.py
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

SEED = 42
HIDDEN_SIZE = 32
DROPOUT = 0.1
EPOCHS = 70
BATCH_SIZE = 32
LR = 1e-3
LOW_RUL_THRESHOLD = 50.0
LOW_RUL_WEIGHT = 3.0


def weighted_mse_loss(preds, targets, weight_mask, low_weight):
    weights = torch.where(weight_mask, torch.full_like(targets, low_weight), torch.ones_like(targets))
    return (weights * (preds - targets) ** 2).mean()


def train_variant(X_train, y_train, train_flight_ids, use_weighting):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    X_train_boot, y_train_boot = bootstrap_by_flight(X_train, y_train, train_flight_ids, SEED)
    train_loader = DataLoader(RULDataset(X_train_boot, y_train_boot), batch_size=BATCH_SIZE, shuffle=True)
    model = RULRegressorVariant(hidden_size=HIDDEN_SIZE, dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            if use_weighting:
                mask = yb < LOW_RUL_THRESHOLD
                loss = weighted_mse_loss(preds, yb, mask, LOW_RUL_WEIGHT)
            else:
                loss = nn.functional.mse_loss(preds, yb)
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

    print("Training baseline ...")
    model_base = train_variant(X_train_scaled, y_train, train_flight_ids_arr, use_weighting=False)
    print("Training reweighted ...")
    model_weighted = train_variant(X_train_scaled, y_train, train_flight_ids_arr, use_weighting=True)

    preds_b = predict_all(model_base, X_val_scaled)
    preds_w = predict_all(model_weighted, X_val_scaled)

    low_mask = y_val < LOW_RUL_THRESHOLD

    # --- Per-flight breakdown, bucket<50 windows only ---
    print("\n=== Per-flight breakdown (bucket<50 windows only) ===")
    print(f"{'flight':>8} {'n':>4} {'base_RMSE':>10} {'rewt_RMSE':>10} {'base_bias':>10} {'rewt_bias':>10}")
    by_flight = defaultdict(list)
    low_idx = np.where(low_mask)[0]
    for i in low_idx:
        by_flight[val_flight_ids[i]].append(i)

    for fid, idxs in sorted(by_flight.items()):
        idxs = np.array(idxs)
        true_v = y_val[idxs]
        b = preds_b[idxs]
        w = preds_w[idxs]
        base_rmse = np.sqrt(np.mean((b - true_v) ** 2))
        rewt_rmse = np.sqrt(np.mean((w - true_v) ** 2))
        base_bias = np.mean(b - true_v)
        rewt_bias = np.mean(w - true_v)
        print(f"{fid:>8} {len(idxs):>4} {base_rmse:>10.2f} {rewt_rmse:>10.2f} {base_bias:>+10.2f} {rewt_bias:>+10.2f}")

    # --- Negative-prediction check ---
    print("\n=== Negative prediction counts ===")
    print(f"  bucket<50 val windows: {np.sum(low_mask)}")
    print(f"    baseline negative preds:   {np.sum(preds_b[low_mask] < 0)}")
    print(f"    reweighted negative preds: {np.sum(preds_w[low_mask] < 0)}")
    print(f"  ALL val windows: {len(y_val)}")
    print(f"    baseline negative preds:   {np.sum(preds_b < 0)}")
    print(f"    reweighted negative preds: {np.sum(preds_w < 0)}")
    if np.sum(preds_w < 0) > 0:
        neg_idx = np.where(preds_w < 0)[0]
        neg_true = y_val[neg_idx]
        print(f"    reweighted negative preds -- true_rul range for those: "
              f"min={neg_true.min():.1f} max={neg_true.max():.1f} "
              f"(would be concerning if this extends much above ~10-20)")


if __name__ == "__main__":
    main()