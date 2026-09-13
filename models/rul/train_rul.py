"""
train_rul.py -- RUL (Remaining Useful Life) regressor for AeroTwin.

Built by Ashmitha, standing in for Aashita's Day 2 roadmap item while she
rests, since Ashmitha's own RUL confidence-bound ensemble is blocked on a
first working RUL model existing. This is a first-pass version -- rough
accuracy is fine to start, per the roadmap ("get a first trained model
running end to end before optimizing").

Pipeline:
  1. Load all flights from data/raw/*.csv
  2. Window each flight into overlapping windows (window_size=30, stride=5)
  3. Split by FLIGHT (not by row) into train/test -- prevents leakage
  4. Train a small 1-2 layer LSTM (<50k params) on windowed data
  5. Report RMSE on the held-out test set
  6. Save the trained model + expose predict_rul(window) for teammates

Run: python models/rul/train_rul.py   (from project root, AeroTwin/)
"""
import os
import sys
import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from schema import SENSOR_FIELDS, WINDOW_SIZE, STRIDE

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "raw")
MODEL_OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cpu")  # small model, CPU is fine and keeps this portable


# ---------------------------------------------------------------------------
# 1. Load flights
# ---------------------------------------------------------------------------

def load_all_flights(data_dir=DATA_DIR):
    """
    Loads every CSV in data/raw/ into one list of (flight_id, DataFrame) pairs.
    Skips healthy-only flights for RUL training purposes (their RUL is a
    placeholder -1, not meaningful -- see generate_healthy_dataset).
    """
    csv_paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_paths:
        raise FileNotFoundError(
            f"No CSVs found in {data_dir} -- run simulator/fault_injectors.py first."
        )

    flights = []
    for path in csv_paths:
        df = pd.read_csv(path)
        if df["true_rul_timesteps"].iloc[0] == -1.0 and df["fault_type"].iloc[0] == "none":
            continue  # skip pure-healthy flights, no meaningful RUL label
        flights.append((df["flight_id"].iloc[0], df))
    return flights


# ---------------------------------------------------------------------------
# 2. Windowing
# ---------------------------------------------------------------------------

def create_windows(df, window_size=WINDOW_SIZE, stride=STRIDE):
    """
    Slices one flight's DataFrame into overlapping windows over SENSOR_FIELDS.
    Each window's label is true_rul_timesteps at the window's LAST timestep
    (i.e. "given the last window_size readings, how much time is left").

    Returns (windows, labels):
      windows: np.array shape (n_windows, window_size, n_sensors)
      labels:  np.array shape (n_windows,)
    """
    sensor_data = df[SENSOR_FIELDS].values
    rul = df["true_rul_timesteps"].values

    windows, labels = [], []
    for start in range(0, len(df) - window_size + 1, stride):
        end = start + window_size
        windows.append(sensor_data[start:end])
        labels.append(rul[end - 1])  # label = RUL at the end of the window

    return np.array(windows), np.array(labels)


def build_windowed_dataset(flights):
    """
    Applies create_windows to every flight and concatenates results, while
    keeping track of which flight each window came from -- needed to split
    by flight, not by row, in the next step.
    """
    all_windows, all_labels, all_flight_ids = [], [], []
    for flight_id, df in flights:
        windows, labels = create_windows(df)
        if len(windows) == 0:
            continue  # flight shorter than window_size, skip
        all_windows.append(windows)
        all_labels.append(labels)
        all_flight_ids.extend([flight_id] * len(windows))

    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    flight_ids = np.array(all_flight_ids)
    return X, y, flight_ids


def split_by_flight(X, y, flight_ids, test_size=0.2, seed=42):
    """
    Splits into train/test by FLIGHT ID, not by row -- critical, since
    windows from the same flight are highly correlated and would leak
    information between train and test if split randomly by row.
    """
    unique_flights = np.unique(flight_ids)
    train_flights, test_flights = train_test_split(
        unique_flights, test_size=test_size, random_state=seed
    )
    train_mask = np.isin(flight_ids, train_flights)
    test_mask = np.isin(flight_ids, test_flights)
    return X[train_mask], y[train_mask], X[test_mask], y[test_mask]


# ---------------------------------------------------------------------------
# 3. Model
# ---------------------------------------------------------------------------

class RULDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RULRegressor(nn.Module):
    """
    Small 1-layer LSTM regressor. Kept under 50k params so it trains in
    minutes on CPU, per the roadmap's constraint.
    """
    def __init__(self, n_sensors=len(SENSOR_FIELDS), hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_sensors, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)      # h_n: (1, batch, hidden_size)
        out = self.fc(h_n[-1])          # (batch, 1)
        return out.squeeze(-1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------

def train_model(X_train, y_train, X_test, y_test, epochs=30, batch_size=32, lr=1e-3):
    train_loader = DataLoader(RULDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(RULDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    model = RULRegressor().to(DEVICE)
    n_params = count_params(model)
    print(f"Model has {n_params:,} trainable params (target: <50,000)")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(train_loader.dataset)

        if epoch % 5 == 0 or epoch == epochs:
            test_rmse = evaluate(model, test_loader)
            print(f"Epoch {epoch:3d}/{epochs} -- train MSE: {epoch_loss:8.2f} -- test RMSE: {test_rmse:8.2f}")

    return model


def evaluate(model, loader):
    model.eval()
    total_sq_err, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            total_sq_err += ((preds - yb) ** 2).sum().item()
            n += len(xb)
    return (total_sq_err / n) ** 0.5


# ---------------------------------------------------------------------------
# 5. predict_rul() -- shared function for teammates (Gayatri, Dhruthi, Chinmay)
# ---------------------------------------------------------------------------

def predict_rul(window, model, scaler):
    """
    window: np.array shape (window_size, n_sensors), raw sensor values.
    Returns: point estimate in timesteps (float).

    NOTE: this is the single-model point estimate only. The confidence-bound
    ensemble (lower_bound, recovery_flag) is Ashmitha's next piece, built on
    top of this once multiple model variants exist.
    """
    model.eval()
    scaled = scaler.transform(window.reshape(-1, window.shape[-1])).reshape(window.shape)
    x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)  # add batch dim
    with torch.no_grad():
        pred = model(x).item()
    return max(0.0, pred)  # RUL can't be negative


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading flights from data/raw/ ...")
    flights = load_all_flights()
    print(f"Loaded {len(flights)} fault flights (healthy-only flights excluded).")

    print("Building windowed dataset ...")
    X, y, flight_ids = build_windowed_dataset(flights)
    print(f"Total windows: {len(X)} (window_size={WINDOW_SIZE}, stride={STRIDE})")

    print("Splitting by flight (not by row) ...")
    X_train, y_train, X_test, y_test = split_by_flight(X, y, flight_ids)
    print(f"Train windows: {len(X_train)} -- Test windows: {len(X_test)}")

    print("Scaling sensor features ...")
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    X_train_scaled = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    print("Training RUL regressor ...")
    model = train_model(X_train_scaled, y_train, X_test_scaled, y_test, epochs=70)
    final_rmse = evaluate(model, DataLoader(RULDataset(X_test_scaled, y_test), batch_size=32))
    print(f"\nFinal held-out test RMSE: {final_rmse:.2f} timesteps")

    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_OUT_DIR, "rul_model.pt")
    scaler_path = os.path.join(MODEL_OUT_DIR, "rul_scaler.joblib")
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved model to {model_path}")
    print(f"Saved scaler to {scaler_path}")
    print("\nThis is a first working end-to-end version -- per the roadmap, generate")
    print("more flights for any fault type where test RMSE looks weak, and iterate.")