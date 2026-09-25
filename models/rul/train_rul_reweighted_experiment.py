"""
train_rul_reweighted_experiment.py

Controlled single-variant experiment: does upweighting the loss for
true_rul < 50 windows fix the bucket-(0,50) point-estimate bias (17
val windows where point_est is 30-40+ units above true_rul, confirmed
NOT separable by any hand-engineered feature -- see check_cht_slope_threshold.py
and check_2d_slope_separation.py)?

Matches the v6 investigation's controlled-experiment style: fixed
hyperparameters, single variant, compare against an unweighted baseline
run with the SAME seed/architecture so the only difference is the loss
weighting. Val-only. Does NOT touch test, does NOT modify train_rul_ensemble.py.

Run: python models\\rul\\train_rul_reweighted_experiment.py
"""
import os
import sys

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
    count_params,
)

# Fixed, matched to v6 investigation's controlled experiments
SEED = 42
HIDDEN_SIZE = 32
DROPOUT = 0.1
EPOCHS = 70
BATCH_SIZE = 32
LR = 1e-3
LOW_RUL_THRESHOLD = 50.0
LOW_RUL_WEIGHT = 3.0  # matches v6's "3x weighted loss" precedent


def weighted_mse_loss(preds, targets, weight_mask, low_weight):
    weights = torch.where(weight_mask, torch.full_like(targets, low_weight), torch.ones_like(targets))
    return (weights * (preds - targets) ** 2).mean()


def train_variant(X_train, y_train, train_flight_ids, X_eval, y_eval, use_weighting):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_train_boot, y_train_boot = bootstrap_by_flight(X_train, y_train, train_flight_ids, SEED)

    train_loader = DataLoader(RULDataset(X_train_boot, y_train_boot), batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(RULDataset(X_eval, y_eval), batch_size=BATCH_SIZE, shuffle=False)

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

    return model, eval_loader


def evaluate_bucket0(model, X_eval, y_eval):
    """Reports RMSE overall and specifically on true_rul < 50, plus mean
    signed error on that subset (to see if bias direction/magnitude improves,
    not just whether coverage passes)."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(X_eval)):
            x = torch.tensor(X_eval[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            preds.append(model(x).item())
    preds = np.array(preds)

    overall_rmse = np.sqrt(np.mean((preds - y_eval) ** 2))

    low_mask = y_eval < 50
    low_rmse = np.sqrt(np.mean((preds[low_mask] - y_eval[low_mask]) ** 2))
    low_bias = np.mean(preds[low_mask] - y_eval[low_mask])  # positive = overshooting, as seen before

    return overall_rmse, low_rmse, low_bias, preds


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
    train_flight_ids = flight_ids[train_mask]

    print("Scaling (fit on train only, same as production pipeline) ...")
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    X_train_scaled = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

    print(f"\nTrain windows: {len(X_train)}  Val windows: {len(X_val)}  "
          f"(bucket 0-50 val windows: {np.sum(y_val < 50)})")

    print("\n=== Experiment 1: BASELINE (unweighted MSE, matched hyperparameters) ===")
    model_base, eval_loader = train_variant(
        X_train_scaled, y_train, train_flight_ids, X_val_scaled, y_val, use_weighting=False
    )
    rmse_b, low_rmse_b, low_bias_b, preds_b = evaluate_bucket0(model_base, X_val_scaled, y_val)
    print(f"  overall RMSE: {rmse_b:.2f}")
    print(f"  bucket<50 RMSE: {low_rmse_b:.2f}")
    print(f"  bucket<50 mean signed error (point_est - true): {low_bias_b:+.2f}")

    print(f"\n=== Experiment 2: REWEIGHTED ({LOW_RUL_WEIGHT}x loss on true_rul < {LOW_RUL_THRESHOLD}) ===")
    model_weighted, _ = train_variant(
        X_train_scaled, y_train, train_flight_ids, X_val_scaled, y_val, use_weighting=True
    )
    rmse_w, low_rmse_w, low_bias_w, preds_w = evaluate_bucket0(model_weighted, X_val_scaled, y_val)
    print(f"  overall RMSE: {rmse_w:.2f}")
    print(f"  bucket<50 RMSE: {low_rmse_w:.2f}")
    print(f"  bucket<50 mean signed error (point_est - true): {low_bias_w:+.2f}")

    print("\n=== Comparison ===")
    print(f"  overall RMSE:      baseline={rmse_b:.2f}   reweighted={rmse_w:.2f}   "
          f"({'better' if rmse_w < rmse_b else 'worse'})")
    print(f"  bucket<50 RMSE:    baseline={low_rmse_b:.2f}   reweighted={low_rmse_w:.2f}   "
          f"({'better' if low_rmse_w < low_rmse_b else 'worse'})")
    print(f"  bucket<50 bias:    baseline={low_bias_b:+.2f}  reweighted={low_bias_w:+.2f}  "
          f"({'reduced' if abs(low_bias_w) < abs(low_bias_b) else 'NOT reduced'})")

    # Specifically re-check the 17 known windows from earlier sessions
    print("\n=== The 17 previously-known failing windows (val split), baseline vs reweighted ===")
    low_idx = np.where(y_val < 50)[0]
    for i in low_idx:
        # only print ones that were near-miss/failing before (point_est was 30+ over true)
        if preds_b[i] - y_val[i] > 20:
            print(f"  idx={i:4d}  true={y_val[i]:6.1f}  "
                  f"baseline_pred={preds_b[i]:7.1f}  reweighted_pred={preds_w[i]:7.1f}")

    print("\nNOTE: this is a single-variant experiment, not the full ensemble, and models")
    print("are NOT saved -- purely to check whether reweighting moves the needle before")
    print("committing to retraining the full N_VARIANTS ensemble in train_rul_ensemble.py.")
    print("If bucket<50 RMSE/bias improve meaningfully here, next step is adding the same")
    print("weighted-loss logic into train_one_variant() for all variants and re-fitting")
    print("calibration (fit_calibration) on the new ensemble's validation predictions.")


if __name__ == "__main__":
    main()