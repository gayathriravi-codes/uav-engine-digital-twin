"""
train_rul_model.py — trains a small LSTM regressor to predict remaining
useful life (RUL, in timesteps) from a windowed sensor sequence.

Excludes healthy-only flights from training (their true_rul is a -1
sentinel, not a real countdown -- they're for the classifier's "none"
class, not for RUL regression).

Run directly: python simulator/train_rul_model.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from preprocessing import build_dataset, train_test_split_by_flight
from schema import SENSOR_FIELDS, WINDOW_SIZE


def filter_rul_windows(windows):
    """Drop windows with no meaningful RUL label (healthy-only flights)."""
    return [w for w in windows if w["true_rul"] >= 0]


class WindowDataset(Dataset):
    def __init__(self, windows, feature_mean, feature_std):
        self.windows = windows
        self.mean = feature_mean
        self.std = feature_std

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        x = (w["features"] - self.mean) / self.std
        y = w["true_rul"]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def compute_norm_stats(windows):
    """Mean/std per sensor feature, computed ONLY on training windows."""
    all_features = np.concatenate([w["features"] for w in windows], axis=0)
    return all_features.mean(axis=0), all_features.std(axis=0) + 1e-6


class RULRegressor(nn.Module):
    """Small LSTM regressor -- kept under 50k params so it trains fast on CPU."""
    def __init__(self, n_features=len(SENSOR_FIELDS), hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)          # h_n: (num_layers, batch, hidden)
        last_hidden = h_n[-1]                # (batch, hidden)
        return self.fc(last_hidden).squeeze(-1)  # (batch,)


def train_model(train_loader, n_features, epochs=15, lr=1e-3):
    model = RULRegressor(n_features=n_features)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} -- train MSE: {avg_loss:.2f}")
    return model


def evaluate_rmse(model, test_loader):
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            pred = model(x_batch)
            all_preds.append(pred.numpy())
            all_true.append(y_batch.numpy())
    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)
    rmse = np.sqrt(np.mean((all_preds - all_true) ** 2))
    return rmse, all_preds, all_true


if __name__ == "__main__":
    print("Loading and windowing data...")
    all_windows, flight_ids = build_dataset("data/raw")
    train_w, test_w = train_test_split_by_flight(all_windows, flight_ids)

    train_w = filter_rul_windows(train_w)
    test_w = filter_rul_windows(test_w)
    print(f"RUL training windows: {len(train_w)}  |  RUL test windows: {len(test_w)}")

    if len(train_w) == 0 or len(test_w) == 0:
        print("Not enough data to train -- check that data/raw has fault flights generated.")
        sys.exit(1)

    feature_mean, feature_std = compute_norm_stats(train_w)

    train_ds = WindowDataset(train_w, feature_mean, feature_std)
    test_ds = WindowDataset(test_w, feature_mean, feature_std)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    print("\nTraining RUL regressor...")
    model = train_model(train_loader, n_features=len(SENSOR_FIELDS))

    rmse, preds, true = evaluate_rmse(model, test_loader)
    print(f"\n=== Held-out test RMSE: {rmse:.2f} timesteps ===")
    print(f"True RUL range in test set: {true.min():.0f} to {true.max():.0f}")
    print(f"Example predictions vs true (first 5): "
          f"{list(zip(np.round(preds[:5], 1), true[:5]))}")

    os.makedirs("models/rul", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
    }, "models/rul/rul_model.pt")
    print("\nSaved model to models/rul/rul_model.pt")