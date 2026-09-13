"""
experiment_elapsed_feature_v2.py -- ONE-OFF experiment, not production code.

Re-tests whether adding elapsed_fraction helps the RUL>=250 bucket, this
time using the REAL train_one_variant() from train_rul_ensemble.py (same
architecture, dropout, bootstrapping) instead of a stripped-down model --
the v1 experiment's quick model collapsed to predicting a constant (std=0.0),
which wasn't a fair test of the feature itself.

Trains ONE real variant (hidden_size=32, dropout=0.1, matching a mid-range
setting from HIDDEN_SIZES/DROPOUTS in train_rul_ensemble.py) with 11
features (7 raw + 3 derived + elapsed_fraction) vs the same setup WITHOUT
elapsed_fraction, and compares raw prediction spread for true RUL>=250.
"""
import sys, os
sys.path.insert(0, 'models/rul')
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from train_rul_ensemble import (
    load_all_flights, build_inference_window, build_windowed_dataset_v5,
    train_one_variant, WINDOW_SIZE, STRIDE, RULRegressorVariant, DEVICE,
)
from schema import SENSOR_FIELDS


def create_windows_with_elapsed(df, window_size=WINDOW_SIZE, stride=STRIDE):
    sensor_data = df[SENSOR_FIELDS].values
    rul = df["true_rul_timesteps"].values
    flight_len = len(df)

    windows, labels = [], []
    for start in range(0, len(df) - window_size + 1, stride):
        end = start + window_size
        raw_window = sensor_data[start:end]
        full_window = build_inference_window(raw_window)
        elapsed_frac = end / flight_len
        elapsed_col = np.full((window_size, 1), elapsed_frac)
        window_with_elapsed = np.concatenate([full_window, elapsed_col], axis=1)
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


def predict_raw_single_model(model, X, scaler):
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

    # --- Dataset WITHOUT elapsed_fraction (10 features, matches real pipeline) ---
    print("Building baseline dataset (10 features, no elapsed_fraction) ...")
    X10, y, flight_ids = build_windowed_dataset_v5(flights)

    # --- Dataset WITH elapsed_fraction (11 features) ---
    print("Building dataset WITH elapsed_fraction (11 features) ...")
    X11, y11, flight_ids11 = build_dataset_with_elapsed(flights)
    assert np.array_equal(y, y11), "Label mismatch between the two datasets -- something's wrong."

    unique_flights = np.unique(flight_ids)
    train_flights, temp_flights = train_test_split(unique_flights, test_size=0.3, random_state=42)
    val_flights, test_flights = train_test_split(temp_flights, test_size=0.5, random_state=42)

    train_mask = np.isin(flight_ids, train_flights)
    val_mask = np.isin(flight_ids, val_flights)
    train_flight_ids = flight_ids[train_mask]

    # ---------- Baseline (10 features) ----------
    X10_train, y_train = X10[train_mask], y[train_mask]
    X10_val, y_val = X10[val_mask], y[val_mask]

    scaler10 = StandardScaler()
    scaler10.fit(X10_train.reshape(-1, X10_train.shape[-1]))
    X10_train_scaled = scaler10.transform(X10_train.reshape(-1, X10_train.shape[-1])).reshape(X10_train.shape)
    X10_val_scaled = scaler10.transform(X10_val.reshape(-1, X10_val.shape[-1])).reshape(X10_val.shape)

    print("\nTraining REAL variant WITHOUT elapsed_fraction (baseline, 10 features) ...")
    model10, rmse10, dropped_idx10, _ = train_one_variant(
        X10_train_scaled, y_train, train_flight_ids, X10_val_scaled, y_val,
        seed=0, dropout=0.1, hidden_size=32, n_dropped_features=0
    )
    print(f"  val RMSE (baseline, 10 features): {rmse10:.2f}")

    target_mask = (y_val >= 250) & (y_val < 350)
    print(f"  True windows in (250,350): {target_mask.sum()}")
    raw_preds10 = predict_raw_single_model(model10, X10_val[target_mask], scaler10)
    print(f"  BASELINE raw preds: min={raw_preds10.min():.1f}  max={raw_preds10.max():.1f}  "
          f"mean={raw_preds10.mean():.1f}  std={raw_preds10.std():.1f}")
    print(f"  How many land in (250,350)? {((raw_preds10>=250)&(raw_preds10<350)).sum()} / {len(raw_preds10)}")

    # ---------- With elapsed_fraction (11 features) ----------
    X11_train, y_train11 = X11[train_mask], y11[train_mask]
    X11_val, y_val11 = X11[val_mask], y11[val_mask]

    scaler11 = StandardScaler()
    scaler11.fit(X11_train.reshape(-1, X11_train.shape[-1]))
    X11_train_scaled = scaler11.transform(X11_train.reshape(-1, X11_train.shape[-1])).reshape(X11_train.shape)
    X11_val_scaled = scaler11.transform(X11_val.reshape(-1, X11_val.shape[-1])).reshape(X11_val.shape)

    # train_one_variant builds RULRegressorVariant internally with a fixed
    # n_sensors=len(SENSOR_FIELDS)+3 (10) -- need our own 11-feature model
    # since train_one_variant's internals assume 10. Reimplementing just the
    # training loop here for the 11-feature case, same hyperparams.
    print("\nTraining REAL-STYLE variant WITH elapsed_fraction (11 features) ...")
    torch.manual_seed(0)
    np.random.seed(0)

    from train_rul_ensemble import bootstrap_by_flight, apply_feature_bagging
    from train_rul import RULDataset, evaluate
    from torch.utils.data import DataLoader
    import torch.nn as nn

    X_train_boot, y_train_boot = bootstrap_by_flight(X11_train_scaled, y_train11, train_flight_ids, seed=0)

    train_loader = DataLoader(RULDataset(X_train_boot, y_train_boot), batch_size=32, shuffle=True)
    val_loader = DataLoader(RULDataset(X11_val_scaled, y_val11), batch_size=32, shuffle=False)

    model11 = RULRegressorVariant(n_sensors=11, hidden_size=32, dropout=0.1).to(DEVICE)
    optimizer = torch.optim.Adam(model11.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(70):  # match train_one_variant's default epochs
        model11.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model11(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

    rmse11 = evaluate(model11, val_loader)
    print(f"  val RMSE (with elapsed_fraction, 11 features): {rmse11:.2f}")

    raw_preds11 = predict_raw_single_model(model11, X11_val[target_mask], scaler11)
    print(f"  WITH elapsed_fraction raw preds: min={raw_preds11.min():.1f}  max={raw_preds11.max():.1f}  "
          f"mean={raw_preds11.mean():.1f}  std={raw_preds11.std():.1f}")
    print(f"  How many land in (250,350)? {((raw_preds11>=250)&(raw_preds11<350)).sum()} / {len(raw_preds11)}")

    print("\n=== SUMMARY ===")
    print(f"Baseline (10 feat):          RMSE={rmse10:.2f}  RUL>=250 std={raw_preds10.std():.1f}  "
          f"mean={raw_preds10.mean():.1f}  correct_bucket={((raw_preds10>=250)&(raw_preds10<350)).sum()}/{len(raw_preds10)}")
    print(f"With elapsed_fraction (11 feat): RMSE={rmse11:.2f}  RUL>=250 std={raw_preds11.std():.1f}  "
          f"mean={raw_preds11.mean():.1f}  correct_bucket={((raw_preds11>=250)&(raw_preds11<350)).sum()}/{len(raw_preds11)}")