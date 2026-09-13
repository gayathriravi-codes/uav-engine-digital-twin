"""
experiment_weighted_loss.py -- ONE-OFF experiment, not production code.

Tests whether weighting high-RUL windows more heavily in the loss function
fixes the (250,350) bucket collapse that persisted even after adding
elapsed_fraction (see experiment_elapsed_feature_v2.py -- RMSE improved
37.35->15.92 overall, but RUL>=250 group still collapsed to std=0.0).

Trains THREE comparable single variants (same seed/dropout/hidden_size/
bootstrapping as train_one_variant(), 70 epochs each):
  1. baseline: 10 features, unweighted MSE loss
  2. + elapsed_fraction: 11 features, unweighted MSE loss
  3. + elapsed_fraction + weighted loss: 11 features, RUL>=250 windows
     weighted WEIGHT_HIGH_RUL x more heavily than others

WEIGHT_HIGH_RUL is set to roughly the inverse of the class's training
frequency (9.1% of windows -> weight ~3x is a moderate starting point,
not full inverse-frequency 1/0.091=11x, to avoid destabilizing training
on the majority class).
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
    load_all_flights, build_windowed_dataset_v5, _compute_derived_features,
    train_one_variant, bootstrap_by_flight, RULRegressorVariant, WINDOW_SIZE, STRIDE, DEVICE,
)
from train_rul import RULDataset, evaluate
from schema import SENSOR_FIELDS

WEIGHT_HIGH_RUL = 3.0  # tunable -- how much more heavily RUL>=250 windows count in the loss
HIGH_RUL_THRESHOLD = 250


def build_inference_window_experimental(raw_window, elapsed_fraction):
    """
    Experiment-only copy of build_inference_window, extended with
    elapsed_fraction. Does NOT modify the production function in
    train_rul_ensemble.py -- kept fully isolated until this recipe
    (feature + weighted loss) is validated.
    """
    derived = _compute_derived_features(raw_window)
    derived_broadcast = np.tile(derived, (raw_window.shape[0], 1))
    elapsed_col = np.full((raw_window.shape[0], 1), elapsed_fraction)
    return np.concatenate([raw_window, derived_broadcast, elapsed_col], axis=1)


def create_windows_with_elapsed(df, window_size=WINDOW_SIZE, stride=STRIDE):
    sensor_data = df[SENSOR_FIELDS].values
    rul = df["true_rul_timesteps"].values
    flight_len = len(df)

    windows, labels = [], []
    for start in range(0, len(df) - window_size + 1, stride):
        end = start + window_size
        raw_window = sensor_data[start:end]
        elapsed_frac = end / flight_len
        full_window = build_inference_window_experimental(raw_window, elapsed_frac)
        windows.append(full_window)
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


def train_weighted_variant(X_train, y_train, train_flight_ids, X_eval, y_eval,
                            n_sensors, weight_high_rul, high_rul_threshold,
                            seed=0, dropout=0.1, hidden_size=32, epochs=70, batch_size=32, lr=1e-3):
    """
    Same recipe as train_one_variant(), but with per-window loss weighting:
    windows with true RUL >= high_rul_threshold count weight_high_rul x
    more heavily in the loss than everything else.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train_boot, y_train_boot = bootstrap_by_flight(X_train, y_train, train_flight_ids, seed)

    train_loader = DataLoader(RULDataset(X_train_boot, y_train_boot), batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(RULDataset(X_eval, y_eval), batch_size=batch_size, shuffle=False)

    model = RULRegressorVariant(n_sensors=n_sensors, hidden_size=hidden_size, dropout=dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss(reduction='none')  # per-element, so we can weight before averaging

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            per_window_loss = loss_fn(preds, yb)
            weights = torch.where(yb >= high_rul_threshold,
                                   torch.tensor(weight_high_rul, device=DEVICE),
                                   torch.tensor(1.0, device=DEVICE))
            loss = (per_window_loss * weights).mean()
            loss.backward()
            optimizer.step()

    eval_rmse = evaluate(model, eval_loader)
    return model, eval_rmse


if __name__ == "__main__":
    print("Loading flights ...")
    flights = load_all_flights()

    print("Building baseline dataset (10 features) ...")
    X10, y, flight_ids = build_windowed_dataset_v5(flights)

    print("Building dataset WITH elapsed_fraction (11 features) ...")
    X11, y11, flight_ids11 = build_dataset_with_elapsed(flights)
    assert np.array_equal(y, y11), "Label mismatch -- something's wrong."

    unique_flights = np.unique(flight_ids)
    train_flights, temp_flights = train_test_split(unique_flights, test_size=0.3, random_state=42)
    val_flights, test_flights = train_test_split(temp_flights, test_size=0.5, random_state=42)

    train_mask = np.isin(flight_ids, train_flights)
    val_mask = np.isin(flight_ids, val_flights)
    train_flight_ids = flight_ids[train_mask]

    target_mask_full = None  # set after we know y_val

    def summarize(name, model, X_val_raw, y_val, scaler, rmse):
        global target_mask_full
        target_mask_full = (y_val >= 250) & (y_val < 350)
        raw_preds = predict_raw_single_model(model, X_val_raw[target_mask_full], scaler)
        correct = ((raw_preds >= 250) & (raw_preds < 350)).sum()
        print(f"\n[{name}] val RMSE={rmse:.2f}  RUL>=250 group (n={target_mask_full.sum()}):")
        print(f"    raw preds: min={raw_preds.min():.1f}  max={raw_preds.max():.1f}  "
              f"mean={raw_preds.mean():.1f}  std={raw_preds.std():.1f}  correct_bucket={correct}/{len(raw_preds)}")
        return rmse, raw_preds

    # ---------- 1. Baseline: 10 features, unweighted ----------
    X10_train, y_train = X10[train_mask], y[train_mask]
    X10_val, y_val = X10[val_mask], y[val_mask]
    scaler10 = StandardScaler()
    scaler10.fit(X10_train.reshape(-1, X10_train.shape[-1]))
    X10_train_scaled = scaler10.transform(X10_train.reshape(-1, X10_train.shape[-1])).reshape(X10_train.shape)
    X10_val_scaled = scaler10.transform(X10_val.reshape(-1, X10_val.shape[-1])).reshape(X10_val.shape)

    print("\n=== 1/3: Baseline (10 features, unweighted) ===")
    model1, rmse1, _, _ = train_one_variant(
        X10_train_scaled, y_train, train_flight_ids, X10_val_scaled, y_val,
        seed=0, dropout=0.1, hidden_size=32, n_dropped_features=0
    )
    summarize("1. Baseline (10 feat, unweighted)", model1, X10_val, y_val, scaler10, rmse1)

    # ---------- 2. + elapsed_fraction, unweighted ----------
    X11_train, y_train11 = X11[train_mask], y11[train_mask]
    X11_val, y_val11 = X11[val_mask], y11[val_mask]
    scaler11 = StandardScaler()
    scaler11.fit(X11_train.reshape(-1, X11_train.shape[-1]))
    X11_train_scaled = scaler11.transform(X11_train.reshape(-1, X11_train.shape[-1])).reshape(X11_train.shape)
    X11_val_scaled = scaler11.transform(X11_val.reshape(-1, X11_val.shape[-1])).reshape(X11_val.shape)

    print("\n=== 2/3: + elapsed_fraction (11 features, unweighted) ===")
    model2, rmse2 = train_weighted_variant(
        X11_train_scaled, y_train11, train_flight_ids, X11_val_scaled, y_val11,
        n_sensors=11, weight_high_rul=1.0, high_rul_threshold=HIGH_RUL_THRESHOLD,  # weight=1.0 == unweighted
        seed=0, dropout=0.1, hidden_size=32
    )
    summarize("2. + elapsed_fraction (11 feat, unweighted)", model2, X11_val, y_val11, scaler11, rmse2)

    # ---------- 3. + elapsed_fraction + weighted loss ----------
    print(f"\n=== 3/3: + elapsed_fraction + weighted loss (weight={WEIGHT_HIGH_RUL}x for RUL>={HIGH_RUL_THRESHOLD}) ===")
    model3, rmse3 = train_weighted_variant(
        X11_train_scaled, y_train11, train_flight_ids, X11_val_scaled, y_val11,
        n_sensors=11, weight_high_rul=WEIGHT_HIGH_RUL, high_rul_threshold=HIGH_RUL_THRESHOLD,
        seed=0, dropout=0.1, hidden_size=32
    )
    summarize("3. + elapsed_fraction + weighted loss", model3, X11_val, y_val11, scaler11, rmse3)

    print("\n=== FINAL COMPARISON ===")
    print(f"1. Baseline:                        RMSE={rmse1:.2f}")
    print(f"2. + elapsed_fraction:               RMSE={rmse2:.2f}")
    print(f"3. + elapsed_fraction + weighted:     RMSE={rmse3:.2f}")
    print("(Check the RUL>=250 std and correct_bucket counts printed above for each --")
    print(" the real question is whether #3's std moved above 0 and correct_bucket > 0.)")