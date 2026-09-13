"""
train_rul_ensemble.py -- RUL confidence-bound ensemble for AeroTwin. (v3)

Built by Ashmitha, on top of Aashita/Ashmitha's train_rul.py.

History:
  v1: seed + dropout variation only -> ensemble std too tight (~1.6-4.2 min)
  v2: added per-variant flight bootstrapping -> RMSEs spread out across
      variants but per-window std barely moved (~2.0-3.0 min). Same
      architecture on resampled data still converges to near-identical
      predictions on a given input.
  v3 (this version): adds two more real sources of diversity on top of v2's
      bootstrapping:
    - hidden_size varies per variant (16/24/32/40/48) -- different capacity
      models learn genuinely different functions, unlike same-architecture
      models trained on resampled data.
    - feature bagging -- each variant randomly drops 1-2 sensor columns
      (zeroed out, not removed, so input shape stays fixed across variants)
      forcing it to rely on a different subset of signals. Classic
      random-forest-style trick for creating real prediction disagreement,
      especially on unusual/edge-case windows.

Trains N variant RULRegressor models, then exposes predict_rul_ensemble()
which returns:
  - point estimate (mean of variant predictions)
  - rul_lower_bound_minutes (MIN of variant predictions, clipped >= 0)
  - std across variants (useful for the dashboard's confidence display)

NOTE: still open from v2 -- point estimates in the sanity check have
clustered noticeably below true RUL on high-true-RUL windows in every run
so far. That looks like a possible systematic underestimation / data
imbalance issue (windows with true_rul > ~230 may be underrepresented in
training), separate from the ensemble-diversity work here. Planned as the
next investigation once this version's spread looks meaningful.

Run: python models/rul/train_rul_ensemble.py   (from project root, AeroTwin/)
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rul import (
    load_all_flights,
    build_windowed_dataset,
    split_by_flight,
    RULDataset,
    evaluate,
    DEVICE,
    MODEL_OUT_DIR,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from schema import SENSOR_FIELDS

N_VARIANTS = 5
SEEDS = [0, 1, 2, 3, 4]
DROPOUTS = [0.0, 0.1, 0.15, 0.2, 0.25]
HIDDEN_SIZES = [16, 24, 32, 40, 48]     # capacity diversity across variants
N_DROPPED_FEATURES = [0, 1, 1, 2, 2]    # how many sensor columns each variant zeroes out


# ---------------------------------------------------------------------------
# Variant model -- same architecture shape as RULRegressor, but hidden_size
# and dropout vary per variant.
# ---------------------------------------------------------------------------

class RULRegressorVariant(nn.Module):
    def __init__(self, n_sensors=len(SENSOR_FIELDS), hidden_size=32, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_sensors, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(self.dropout(h_n[-1]))
        return out.squeeze(-1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def bootstrap_by_flight(X, y, flight_ids, seed):
    """
    Resamples FLIGHTS with replacement (not individual windows/rows), then
    gathers all windows belonging to the resampled flights.
    """
    rng = np.random.RandomState(seed)
    unique_flights = np.unique(flight_ids)
    sampled_flights = rng.choice(unique_flights, size=len(unique_flights), replace=True)

    idx_list = []
    for f in sampled_flights:
        matches = np.where(flight_ids == f)[0]
        idx_list.append(matches)
    idx = np.concatenate(idx_list)

    return X[idx], y[idx]


def apply_feature_bagging(X, n_dropped, seed):
    """
    Zeroes out n_dropped randomly chosen sensor columns (last axis) across
    every window in X. Zeroing (not removing) keeps input shape fixed so
    the same architecture still applies -- the model just can't see those
    sensors for this variant, forcing reliance on the remaining ones.
    Applied identically at train AND inference time for a given variant
    (see predict_rul_ensemble), so the model is self-consistent.
    """
    if n_dropped == 0:
        return X, np.array([], dtype=int)
    rng = np.random.RandomState(seed + 1000)  # offset so it differs from bootstrap's seed use
    n_sensors = X.shape[-1]
    dropped_idx = rng.choice(n_sensors, size=n_dropped, replace=False)
    X_bagged = X.copy()
    X_bagged[:, :, dropped_idx] = 0.0
    return X_bagged, dropped_idx


def train_one_variant(X_train, y_train, train_flight_ids, X_test, y_test,
                       seed, dropout, hidden_size, n_dropped_features,
                       epochs=70, batch_size=32, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train_boot, y_train_boot = bootstrap_by_flight(X_train, y_train, train_flight_ids, seed)
    X_train_boot, dropped_idx = apply_feature_bagging(X_train_boot, n_dropped_features, seed)
    # Apply the SAME dropped features to the test set for a fair/consistent eval
    X_test_bagged = X_test.copy()
    if len(dropped_idx) > 0:
        X_test_bagged[:, :, dropped_idx] = 0.0

    train_loader = DataLoader(RULDataset(X_train_boot, y_train_boot), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(RULDataset(X_test_bagged, y_test), batch_size=batch_size, shuffle=False)

    model = RULRegressorVariant(hidden_size=hidden_size, dropout=dropout).to(DEVICE)
    n_params = count_params(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

    test_rmse = evaluate(model, test_loader)
    return model, test_rmse, dropped_idx, n_params


# ---------------------------------------------------------------------------
# Ensemble inference -- shared function for teammates
# ---------------------------------------------------------------------------

def predict_rul_ensemble(window, models, scaler, dropped_idx_list):
    """
    window: np.array shape (window_size, n_sensors), raw sensor values.
    models: list of trained RULRegressorVariant instances.
    scaler: the StandardScaler fit during ensemble training.
    dropped_idx_list: list of dropped-feature-index arrays, one per model,
        in the SAME order as `models` (returned by training, or reload
        alongside the saved variant weights).

    Returns dict:
      point_estimate_minutes -- mean across variants
      rul_lower_bound_minutes -- min across variants, clipped >= 0
      std_minutes -- spread across variants (for dashboard confidence display)
    """
    scaled = scaler.transform(window.reshape(-1, window.shape[-1])).reshape(window.shape)

    preds = []
    with torch.no_grad():
        for model, dropped_idx in zip(models, dropped_idx_list):
            model.eval()
            window_variant = scaled.copy()
            if len(dropped_idx) > 0:
                window_variant[:, dropped_idx] = 0.0
            x = torch.tensor(window_variant, dtype=torch.float32).unsqueeze(0)
            preds.append(model(x).item())

    preds = np.array(preds)
    point_estimate = max(0.0, float(preds.mean()))
    lower_bound = max(0.0, float(preds.min()))
    std = float(preds.std())

    return {
        "point_estimate_minutes": point_estimate,
        "rul_lower_bound_minutes": lower_bound,
        "std_minutes": std,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading flights from data/raw/ ...")
    flights = load_all_flights()
    print(f"Loaded {len(flights)} fault flights (healthy-only flights excluded).")

    print("Building windowed dataset ...")
    X, y, flight_ids = build_windowed_dataset(flights)
    print(f"Total windows: {len(X)}")

    print("Splitting by flight (not by row) ...")
    X_train, y_train, X_test, y_test = split_by_flight(X, y, flight_ids)
    print(f"Train windows: {len(X_train)} -- Test windows: {len(X_test)}")

    unique_flights = np.unique(flight_ids)
    train_flights, test_flights = train_test_split(unique_flights, test_size=0.2, random_state=42)
    train_mask = np.isin(flight_ids, train_flights)
    train_flight_ids = flight_ids[train_mask]

    print("Scaling sensor features (fit on train only) ...")
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    X_train_scaled = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    print(f"\nTraining {N_VARIANTS} ensemble variants (bootstrapping + hidden_size + feature bagging) ...")
    models = []
    rmses = []
    dropped_idx_list = []
    for i in range(N_VARIANTS):
        seed = SEEDS[i]
        dropout = DROPOUTS[i]
        hidden_size = HIDDEN_SIZES[i]
        n_dropped = N_DROPPED_FEATURES[i]
        print(f"  Variant {i+1}/{N_VARIANTS} -- seed={seed}, dropout={dropout}, "
              f"hidden_size={hidden_size}, n_dropped_features={n_dropped} ...")
        model, rmse, dropped_idx, n_params = train_one_variant(
            X_train_scaled, y_train, train_flight_ids, X_test_scaled, y_test,
            seed, dropout, hidden_size, n_dropped
        )
        dropped_names = [SENSOR_FIELDS[j] for j in dropped_idx]
        print(f"    -> test RMSE: {rmse:.2f}  ({n_params:,} params)  dropped: {dropped_names}")
        models.append(model)
        rmses.append(rmse)
        dropped_idx_list.append(dropped_idx)

    print(f"\nEnsemble RMSEs: {[round(r, 2) for r in rmses]}")
    print(f"Mean single-model RMSE: {np.mean(rmses):.2f}")

    print("\nSanity check on 5 test windows:")
    for i in range(min(5, len(X_test))):
        result = predict_rul_ensemble(X_test[i], models, scaler, dropped_idx_list)
        true_val = y_test[i]
        assert result["rul_lower_bound_minutes"] <= result["point_estimate_minutes"] + 1e-6, \
            "Lower bound exceeded point estimate!"
        assert result["rul_lower_bound_minutes"] >= 0, "Lower bound went negative!"
        print(f"  true={true_val:7.1f}  point_est={result['point_estimate_minutes']:7.1f}  "
              f"lower_bound={result['rul_lower_bound_minutes']:7.1f}  std={result['std_minutes']:6.1f}")

    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    for i, model in enumerate(models):
        model_path = os.path.join(MODEL_OUT_DIR, f"rul_model_variant_{i}.pt")
        torch.save(model.state_dict(), model_path)
    scaler_path = os.path.join(MODEL_OUT_DIR, "rul_ensemble_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    dropped_idx_path = os.path.join(MODEL_OUT_DIR, "rul_ensemble_dropped_idx.joblib")
    joblib.dump(dropped_idx_list, dropped_idx_path)
    print(f"\nSaved {N_VARIANTS} variant models to {MODEL_OUT_DIR}\\rul_model_variant_*.pt")
    print(f"Saved ensemble scaler to {scaler_path}")
    print(f"Saved dropped-feature indices to {dropped_idx_path}")
    print("\nTeammates: call predict_rul_ensemble(window, models, scaler, dropped_idx_list)")
    print("after loading all variant state_dicts (note: hidden_size differs per variant --")
    print("see HIDDEN_SIZES in this file) and the dropped_idx_list joblib file.")