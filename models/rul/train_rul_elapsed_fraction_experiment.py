"""
train_rul_elapsed_fraction_experiment.py

Controlled single-variant experiment: does adding an elapsed_fraction
feature (window's end position / total flight length) help the model
track bucket-(100,200)'s problem trajectories -- the ones whose point
estimate plateaus near 210-232 largely independent of true_rul through
much of that range (confirmed on test base trajectories 000/013, not
explainable by calibration-band width or the OOD gate)?

This mirrors train_rul_reweighted_experiment.py's format: fixed
hyperparameters, single variant, baseline vs candidate feature, same
seed/architecture so elapsed_fraction is the only difference. Uses the
CORRECTED base-trajectory-grouped split (imports base_trajectory_id from
train_rul_ensemble.py -- does not reimplement it).

Reports overall RMSE AND bucket-(100,200)-specific RMSE/bias, since
overall RMSE improving (as it did in the earlier v6 investigation, which
halved it 37.35->15.92) does NOT by itself mean the targeted bucket
improved -- that's exactly what happened last time elapsed_fraction was
tried, on the (250,350) bucket.

Val-only. Does NOT touch test, does NOT modify train_rul_ensemble.py or
any production artifact.

Run: python models\\rul\\train_rul_elapsed_fraction_experiment.py
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
from train_rul import load_all_flights, RULDataset, DEVICE, WINDOW_SIZE, STRIDE
from train_rul_ensemble import (
    _compute_derived_features, base_trajectory_id, bootstrap_by_flight,
    count_params,
)
from schema import SENSOR_FIELDS

SEED = 42
HIDDEN_SIZE = 32
DROPOUT = 0.1
EPOCHS = 70
BATCH_SIZE = 32
LR = 1e-3

BUCKET_LO, BUCKET_HI = 100, 200


# ---------------------------------------------------------------------------
# Build windows WITH elapsed_fraction as an 11th feature, using the same
# per-window construction as create_windows_v5 but tracking each window's
# position within its flight's full length.
# ---------------------------------------------------------------------------

def create_windows_with_elapsed_fraction(df, window_size=WINDOW_SIZE, stride=STRIDE):
    sensor_data = df[SENSOR_FIELDS].values
    rul = df["true_rul_timesteps"].values
    flight_len = len(df)

    windows, labels = [], []
    for start in range(0, len(df) - window_size + 1, stride):
        end = start + window_size
        raw_window = sensor_data[start:end]

        derived = _compute_derived_features(raw_window)
        elapsed_fraction = end / flight_len  # 0.0 at flight start, ~1.0 near end-of-life

        derived_broadcast = np.tile(derived, (window_size, 1))
        elapsed_broadcast = np.full((window_size, 1), elapsed_fraction)
        full_window = np.concatenate([raw_window, derived_broadcast, elapsed_broadcast], axis=1)

        windows.append(full_window)
        labels.append(rul[end - 1])

    return np.array(windows), np.array(labels)


def build_dataset_with_elapsed_fraction(flights):
    all_windows, all_labels, all_flight_ids = [], [], []
    for flight_id, df in flights:
        windows, labels = create_windows_with_elapsed_fraction(df)
        if len(windows) == 0:
            continue
        all_windows.append(windows)
        all_labels.append(labels)
        all_flight_ids.extend([flight_id] * len(windows))

    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y, np.array(all_flight_ids)


def build_dataset_baseline(flights):
    """Same as build_windowed_dataset_v5, reimplemented locally so this
    script is self-contained and both variants share IDENTICAL window
    boundaries (only the feature set differs)."""
    all_windows, all_labels, all_flight_ids = [], [], []
    for flight_id, df in flights:
        sensor_data = df[SENSOR_FIELDS].values
        rul = df["true_rul_timesteps"].values
        windows, labels = [], []
        for start in range(0, len(df) - WINDOW_SIZE + 1, STRIDE):
            end = start + WINDOW_SIZE
            raw_window = sensor_data[start:end]
            derived = _compute_derived_features(raw_window)
            derived_broadcast = np.tile(derived, (WINDOW_SIZE, 1))
            full_window = np.concatenate([raw_window, derived_broadcast], axis=1)
            windows.append(full_window)
            labels.append(rul[end - 1])
        if len(windows) == 0:
            continue
        all_windows.append(np.array(windows))
        all_labels.append(np.array(labels))
        all_flight_ids.extend([flight_id] * len(windows))

    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y, np.array(all_flight_ids)


# ---------------------------------------------------------------------------
# Model variant, sized to n_features dynamically (11 vs 10 input columns)
# ---------------------------------------------------------------------------

class RULRegressorExperimental(nn.Module):
    def __init__(self, n_features, hidden_size=32, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(self.dropout(h_n[-1]))
        return out.squeeze(-1)


def train_variant(X_train, y_train, train_flight_ids, n_features, seed=SEED):
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


def evaluate_all(model, X_eval, y_eval):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(X_eval)):
            x = torch.tensor(X_eval[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            preds.append(model(x).item())
    preds = np.array(preds)

    overall_rmse = np.sqrt(np.mean((preds - y_eval) ** 2))

    bucket_mask = (y_eval >= BUCKET_LO) & (y_eval < BUCKET_HI)
    bucket_rmse = np.sqrt(np.mean((preds[bucket_mask] - y_eval[bucket_mask]) ** 2))
    bucket_bias = np.mean(preds[bucket_mask] - y_eval[bucket_mask])
    bucket_max_overshoot = np.max(preds[bucket_mask] - y_eval[bucket_mask])

    return overall_rmse, bucket_rmse, bucket_bias, bucket_max_overshoot, preds


def main():
    print("Loading flights ...")
    flights = load_all_flights()

    print("Building BASELINE (10-feature) dataset ...")
    X_base, y_base, flight_ids_base = build_dataset_baseline(flights)
    print("Building ELAPSED-FRACTION (11-feature) dataset ...")
    X_ef, y_ef, flight_ids_ef = build_dataset_with_elapsed_fraction(flights)

    assert len(X_base) == len(X_ef), "Window counts differ between variants -- boundary mismatch!"
    assert np.allclose(y_base, y_ef), "Labels differ between variants -- boundary mismatch!"

    # Corrected base-trajectory-grouped split, identical logic to
    # train_rul_ensemble.py / run_coverage_check.py -- imported, not
    # reimplemented (base_trajectory_id itself IS imported above).
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

    print(f"\nTrain windows: {train_mask.sum()}  Val windows: {val_mask.sum()}  "
          f"(bucket {BUCKET_LO}-{BUCKET_HI} val windows: "
          f"{np.sum((y_base[val_mask] >= BUCKET_LO) & (y_base[val_mask] < BUCKET_HI))})")

    results = {}
    for name, X, y in [("BASELINE (10 features)", X_base, y_base),
                        ("ELAPSED_FRACTION (11 features)", X_ef, y_ef)]:
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        print(f"\nScaling ({name}, fit on train only) ...")
        n_features = X_train.shape[-1]
        scaler = StandardScaler()
        scaler.fit(X_train.reshape(-1, n_features))
        X_train_scaled = scaler.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
        X_val_scaled = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)

        print(f"Training ({name}) ...")
        model = train_variant(X_train_scaled, y_train, train_flight_ids, n_features)
        overall_rmse, bucket_rmse, bucket_bias, bucket_max_overshoot, preds = evaluate_all(
            model, X_val_scaled, y_val
        )

        print(f"  overall val RMSE: {overall_rmse:.2f}")
        print(f"  bucket({BUCKET_LO},{BUCKET_HI}) val RMSE: {bucket_rmse:.2f}")
        print(f"  bucket({BUCKET_LO},{BUCKET_HI}) val bias (pred - true): {bucket_bias:+.2f}")
        print(f"  bucket({BUCKET_LO},{BUCKET_HI}) val MAX overshoot: {bucket_max_overshoot:+.2f}")

        results[name] = {
            "overall_rmse": overall_rmse, "bucket_rmse": bucket_rmse,
            "bucket_bias": bucket_bias, "bucket_max_overshoot": bucket_max_overshoot,
        }

    print("\n=== Comparison ===")
    b = results["BASELINE (10 features)"]
    e = results["ELAPSED_FRACTION (11 features)"]
    print(f"  overall RMSE:        baseline={b['overall_rmse']:.2f}   elapsed_fraction={e['overall_rmse']:.2f}")
    print(f"  bucket RMSE:         baseline={b['bucket_rmse']:.2f}   elapsed_fraction={e['bucket_rmse']:.2f}   "
          f"({'better' if e['bucket_rmse'] < b['bucket_rmse'] else 'NOT better'})")
    print(f"  bucket max overshoot: baseline={b['bucket_max_overshoot']:+.2f}   "
          f"elapsed_fraction={e['bucket_max_overshoot']:+.2f}   "
          f"({'reduced' if e['bucket_max_overshoot'] < b['bucket_max_overshoot'] else 'NOT reduced'})")

    print("\nDECIDING METRIC: bucket_max_overshoot is what actually matters for coverage --")
    print("the earlier finding was that no calibration band fit on val (max ~57-60) could")
    print("cover test's ~110 overshoot. If elapsed_fraction doesn't meaningfully shrink")
    print("bucket_max_overshoot here on val too, it's unlikely to fix the test failure,")
    print("regardless of what overall RMSE does -- same lesson as the (250,350) attempt.")
    print("\nThis is a SINGLE-VARIANT experiment, not the full ensemble, and nothing is")
    print("saved. Do not wire this into production until: (1) bucket_max_overshoot")
    print("genuinely shrinks here, (2) a per-flight/per-base breakdown confirms it's not")
    print("a 1-2-flight collapse (same check as the earlier reweighting experiment), (3)")
    print("multi-seed stability check.")


if __name__ == "__main__":
    main()