"""
experiment_elapsed_feature.py -- ONE-OFF experiment, not production code.

Tests whether adding a single elapsed-time-fraction feature helps the
model distinguish true RUL >= 250 windows, which currently collapse to
a narrow prediction range (185-201, std=4.0) regardless of true value --
see diagnose_250_350_bucket.py findings.

Trains ONE quick model (not the full 7-variant ensemble) with 11 features
(7 raw + 3 derived + elapsed_fraction) vs the baseline 10-feature model,
and compares raw prediction spread specifically for the RUL>=250 group.

elapsed_fraction = window_end_idx / flight_length -- legitimate at real
inference too, since you always know how far into a flight you are.
"""
import sys, os
sys.path.insert(0, 'models/rul')
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from train_rul_ensemble import (
    load_all_flights, build_inference_window, WINDOW_SIZE, STRIDE,
)
from train_rul import RULDataset, evaluate, DEVICE
from schema import SENSOR_FIELDS


def create_windows_with_elapsed(df, window_size=WINDOW_SIZE, stride=STRIDE):
    sensor_data = df[SENSOR_FIELDS].values
    rul = df["true_rul_timesteps"].values
    flight_len = len(df)

    windows, labels = [], []
    for start in range(0, len(df) - window_size + 1, stride):
        end = start + window_size
        raw_window = sensor_data[start:end]
        full_window = build_inference_window(raw_window)          # (window_size, 10)
        elapsed_frac = end / flight_len                            # scalar, 0..1
        elapsed_col = np.full((window_size, 1), elapsed_frac)
        window_with_elapsed = np.concatenate([full_window, elapsed_col], axis=1)  # (window_size, 11)
        windows.append(window_with_elapsed)
        labels.append(rul[end - 1])

    return np.array(windows), np.array(labels)


def build_dataset_with_elapsed(flights):
    all_windows, all_labels, all_flight_ids = [], [], []
    for flight_id, df in flights:
        windows, labels = create_windows_with_elapsed(df)
        if len(windows) == 0:
            continue
        all_windows.append(windows)
        all_labels.append(labels)
        all_flight_ids.extend([flight_id] * len(windows))
    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y, np.array(all_flight_ids)


class SimpleRULRegressor(nn.Module):
    def __init__(self, n_features, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).squeeze(-1)


def train_quick_model(X_train, y_train, X_val, y_val, n_features, epochs=40, seed=0):
    torch.manual_seed(seed)
    train_loader = DataLoader(RULDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(RULDataset(X_val, y_val), batch_size=32, shuffle=False)

    model = SimpleRULRegressor(n_features=n_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

    rmse = evaluate(model, val_loader)
    return model, rmse


def predict_raw(model, X, scaler):
    scaled = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(scaled)):
            x = torch.tensor(scaled[i], dtype=torch.float32).unsqueeze(0)
            preds.append(model(x).item())
    return np.array(preds)


if __name__ == "__main__":
    print("Loading flights ...")
    flights = load_all_flights()

    print("Building dataset WITH elapsed_fraction feature (11 features) ...")
    X11, y, flight_ids = build_dataset_with_elapsed(flights)

    unique_flights = np.unique(flight_ids)
    train_flights, temp_flights = train_test_split(unique_flights, test_size=0.3, random_state=42)
    val_flights, test_flights = train_test_split(temp_flights, test_size=0.5, random_state=42)

    train_mask = np.isin(flight_ids, train_flights)
    val_mask = np.isin(flight_ids, val_flights)

    X11_train, y_train = X11[train_mask], y[train_mask]
    X11_val, y_val = X11[val_mask], y[val_mask]

    scaler11 = StandardScaler()
    scaler11.fit(X11_train.reshape(-1, X11_train.shape[-1]))
    X11_train_scaled = scaler11.transform(X11_train.reshape(-1, X11_train.shape[-1])).reshape(X11_train.shape)
    X11_val_scaled = scaler11.transform(X11_val.reshape(-1, X11_val.shape[-1])).reshape(X11_val.shape)
    
    print("Training quick model WITH elapsed_fraction (11 features) ...")
    model_with, rmse_with = train_quick_model(X11_train_scaled, y_train, X11_val_scaled, y_val, n_features=11)
    print(f"  val RMSE (with elapsed_fraction): {rmse_with:.2f}")

    target_mask = (y_val >= 250) & (y_val < 350)
    print(f"\nTrue windows in (250,350): {target_mask.sum()}")

    raw_preds_with = predict_raw(model_with, X11_val[target_mask], scaler11)
    print(f"WITH elapsed_fraction -- raw preds: min={raw_preds_with.min():.1f}  "
          f"max={raw_preds_with.max():.1f}  mean={raw_preds_with.mean():.1f}  std={raw_preds_with.std():.1f}")
    print(f"How many land in (250,350)? {((raw_preds_with>=250)&(raw_preds_with<350)).sum()} / {len(raw_preds_with)}")

    print("\n(Compare against baseline from diagnose_250_350_bucket.py: "
          "min=185.2 max=201.0 mean=197.5 std=4.0, 0/80 landed correctly)")
