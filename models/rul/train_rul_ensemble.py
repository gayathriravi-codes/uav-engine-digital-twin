"""
train_rul_ensemble.py -- RUL confidence-bound ensemble for AeroTwin. (v5)

Built by Ashmitha, on top of Aashita/Ashmitha's train_rul.py.

History:
  v1: seed + dropout variation only -> ensemble std too tight (~1.6-4.2 min)
  v2: added per-variant flight bootstrapping -> RMSEs spread out across
      variants but per-window std barely moved (~2.0-3.0 min).
  v3: added hidden_size variation + feature bagging (random sensor drop) +
      proper train/val/test split. Val RMSE (honest, no leakage): 44.94.
      Found: hidden_size=16 too weak (val RMSE 58.4); dropping egt+oil_pressure
      together hurt badly (val RMSE 47.2) -- confirmed by sensor_importance.py:
      vibration/cht/oil_pressure/rpm are the RUL_CORE_SENSORS (~90% of total
      importance), oil_temp/egt/fuel_flow are minor.
  v4:
    - HIDDEN_SIZES no longer goes below 24 (16 confirmed too weak in v3)
    - feature bagging is now CORE-AWARE: it will never drop two RUL_CORE_SENSORS
      together in the same variant. At most one core sensor may be dropped
      per variant (alone), or any number of non-core sensors.
    - bumped to 7 variants for more ensemble diversity per the tuning plan
  v5 (this version):
    - Step A fix for P1 (point-estimate bias on high-RUL windows): each
      window now carries 3 derived per-window features (EGT slope,
      integrated deviation from HEALTHY_RANGES, in-window vibration
      variance) alongside the 7 raw sensors -- see create_windows_v5.
      Per-window only, not per-flight cumulative, per the time-constrained
      decision. Lives here, not in train_rul.py, so the original
      single-model baseline (7 sensors only) stays untouched.
    - predict_rul_ensemble's OUTPUT shape is unchanged (point_estimate_minutes,
      rul_lower_bound_minutes, std_minutes) so the dashboard doesn't break.
      Its INPUT now needs the 10-column featurized window, not the raw
      7-column one -- use build_inference_window() below to convert.
    - Step B (this edit): predict_rul_ensemble() takes a new `calibrated`
      kwarg (default False). When True, applies the per-bucket bias
      correction + conformal lower bound from calibrate_rul.py before
      returning. Return dict KEYS are unchanged either way -- teammates
      calling predict_rul_ensemble() with no calibrated= arg get identical
      behavior to before this edit. Default stays False until per-bucket
      coverage is verified on the test set (see
      calibrate_rul.check_per_bucket_coverage) -- only flip the default to
      True after every bucket passes.
        - Step B verified: per-bucket coverage on TEST >= 90% for all buckets
      after merging (100,150)/(150,200) into (100,200) and one iteration
      of refit (bucket-by-corrected-estimate, not raw estimate). calibrated
      default flipped to True. KNOWN CAVEAT: (250,350) bucket got 0
      validation windows after re-bucketing -- its bias/band are 0.0, so
      calibrated output there equals the raw (still-biased) estimate. It
      "passes" coverage only because the raw estimate already undershoots.
      Not a real fix for that bucket -- documented, not solved.


Trains N variant RULRegressor models, then exposes predict_rul_ensemble()
which returns:
  - point estimate (mean of variant predictions)
  - rul_lower_bound_minutes (MIN of variant predictions, clipped >= 0)
  - std across variants (useful for the dashboard's confidence display)

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
    RULDataset,
    evaluate,
    DEVICE,
    MODEL_OUT_DIR,
    WINDOW_SIZE,
    STRIDE,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from schema import SENSOR_FIELDS, HEALTHY_RANGES

# ---------------------------------------------------------------------------
# v5: per-window derived features (Step A, fixing P1 -- point-estimate bias
# on high-RUL windows). Deliberately per-window-only, not per-flight
# cumulative, per the time-constrained decision. Lives here, not in
# train_rul.py, so the original single-model baseline is untouched.
# ---------------------------------------------------------------------------

def _compute_derived_features(window):
    """
    window: np.array shape (window_size, n_sensors=7), raw sensor readings
    Returns: np.array shape (3,) -- [egt_slope, integrated_deviation, vib_variance]
    """
    egt_idx = SENSOR_FIELDS.index("egt")
    vib_idx = SENSOR_FIELDS.index("vibration")

    egt_series = window[:, egt_idx]
    egt_slope = (egt_series[-1] - egt_series[0]) / len(egt_series)

    total_deviation = 0.0
    for i, sensor in enumerate(SENSOR_FIELDS):
        lo, hi = HEALTHY_RANGES[sensor]
        midpoint = (lo + hi) / 2
        range_width = hi - lo
        deviation = np.abs(window[:, i] - midpoint) / range_width
        total_deviation += deviation.sum()

    vib_variance = np.var(window[:, vib_idx])

    return np.array([egt_slope, total_deviation, vib_variance])


def build_inference_window(raw_window):
    """
    Shared helper for teammates / dashboard code doing live inference.

    raw_window: np.array shape (window_size, 7) -- just the raw sensor
        readings, in SENSOR_FIELDS order, same as what fault injectors /
        the simulator already produce. No need to know about v5 internals.

    Returns: np.array shape (window_size, 10) -- the 7 raw sensors plus the
        3 derived features, ready to pass into predict_rul_ensemble().
    """
    derived = _compute_derived_features(raw_window)
    derived_broadcast = np.tile(derived, (raw_window.shape[0], 1))
    return np.concatenate([raw_window, derived_broadcast], axis=1)


def create_windows_v5(df, window_size=WINDOW_SIZE, stride=STRIDE):
    sensor_data = df[SENSOR_FIELDS].values
    rul = df["true_rul_timesteps"].values

    windows, labels = [], []
    for start in range(0, len(df) - window_size + 1, stride):
        end = start + window_size
        raw_window = sensor_data[start:end]
        full_window = build_inference_window(raw_window)
        windows.append(full_window)
        labels.append(rul[end - 1])

    return np.array(windows), np.array(labels)


def build_windowed_dataset_v5(flights):
    all_windows, all_labels, all_flight_ids = [], [], []
    for flight_id, df in flights:
        windows, labels = create_windows_v5(df)
        if len(windows) == 0:
            continue
        all_windows.append(windows)
        all_labels.append(labels)
        all_flight_ids.extend([flight_id] * len(windows))

    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y, np.array(all_flight_ids)
# From sensor_importance.py output (RandomForest R^2 = 0.840 on validation):
#   1. vibration    0.2779
#   2. cht          0.2354
#   3. oil_pressure 0.2251
#   4. rpm          0.1623
#   (oil_temp, egt, fuel_flow are non-core, ~0.03-0.04 each)
RUL_CORE_SENSORS = ['vibration', 'cht', 'oil_pressure', 'rpm']
CORE_SENSOR_IDX = [SENSOR_FIELDS.index(s) for s in RUL_CORE_SENSORS]
NON_CORE_SENSOR_IDX = [i for i in range(len(SENSOR_FIELDS)) if i not in CORE_SENSOR_IDX]

N_VARIANTS = 7
SEEDS = [0, 1, 2, 3, 4, 5, 6]
DROPOUTS = [0.0, 0.1, 0.15, 0.1, 0.2, 0.15, 0.25]
HIDDEN_SIZES = [24, 32, 32, 40, 40, 48, 32]     # never below 24 (16 confirmed too weak in v3)
N_DROPPED_FEATURES = [0, 1, 1, 2, 1, 2, 0]      # how many sensor columns each variant zeroes out


# ---------------------------------------------------------------------------
# Variant model -- same architecture shape as RULRegressor, but hidden_size
# and dropout vary per variant.
# ---------------------------------------------------------------------------

class RULRegressorVariant(nn.Module):
    def __init__(self, n_sensors=len(SENSOR_FIELDS) + 3, hidden_size=32, dropout=0.0):
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
    Zeroes out n_dropped sensor columns, CORE-AWARE:
      - at most ONE of RUL_CORE_SENSORS may be dropped, never two or more together
      - the rest of the drops (if n_dropped > 1) come from NON_CORE_SENSOR_IDX
      - if n_dropped == 0, nothing is dropped
    This prevents the v3 problem where dropping two important sensors
    together (e.g. egt+oil_pressure) tanked a variant's accuracy for no
    diversity benefit worth the cost.
    """
    if n_dropped == 0:
        return X, np.array([], dtype=int)

    rng = np.random.RandomState(seed + 1000)  # offset so it differs from bootstrap's seed use

    dropped = []
    # Randomly decide whether this variant's one "allowed" core drop happens,
    # or whether it drops only non-core sensors instead.
    drop_one_core = rng.random() < 0.5  # 50/50 chance to include a core sensor
    if drop_one_core:
        dropped.append(int(rng.choice(CORE_SENSOR_IDX)))
        remaining = n_dropped - 1
    else:
        remaining = n_dropped

    if remaining > 0:
        non_core_pool = [i for i in NON_CORE_SENSOR_IDX if i not in dropped]
        remaining = min(remaining, len(non_core_pool))  # can't drop more than exist
        dropped.extend(rng.choice(non_core_pool, size=remaining, replace=False).tolist())

    dropped_idx = np.array(sorted(dropped))
    X_bagged = X.copy()
    X_bagged[:, :, dropped_idx] = 0.0
    return X_bagged, dropped_idx


def train_one_variant(X_train, y_train, train_flight_ids, X_eval, y_eval,
                       seed, dropout, hidden_size, n_dropped_features,
                       epochs=70, batch_size=32, lr=1e-3):
    """
    X_eval/y_eval is whatever set you want per-variant RMSE reported against
    during tuning -- pass VALIDATION data here, never test, until the final
    locked-in check.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train_boot, y_train_boot = bootstrap_by_flight(X_train, y_train, train_flight_ids, seed)
    X_train_boot, dropped_idx = apply_feature_bagging(X_train_boot, n_dropped_features, seed)
    X_eval_bagged = X_eval.copy()
    if len(dropped_idx) > 0:
        X_eval_bagged[:, :, dropped_idx] = 0.0

    train_loader = DataLoader(RULDataset(X_train_boot, y_train_boot), batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(RULDataset(X_eval_bagged, y_eval), batch_size=batch_size, shuffle=False)

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

    eval_rmse = evaluate(model, eval_loader)
    return model, eval_rmse, dropped_idx, n_params


# ---------------------------------------------------------------------------
# Ensemble inference -- shared function for teammates
# ---------------------------------------------------------------------------

def predict_rul_ensemble(window, models, scaler, dropped_idx_list, calibrated=True):
    """
    window: np.array shape (window_size, 10) -- the FEATURIZED window
        (7 raw sensors + 3 derived features). If you have a raw 7-column
        sensor window instead, call build_inference_window(raw_window)
        first to get this shape.
    models: list of trained RULRegressorVariant instances.
    scaler: the StandardScaler fit during ensemble training.
    dropped_idx_list: list of dropped-feature-index arrays, one per model,
        in the SAME order as `models`.
    calibrated: if True, applies Step B's per-bucket bias correction +
        conformal lower bound (see calibrate_rul.py) before returning.
        Return dict KEYS are unchanged either way -- point_estimate_minutes
        and rul_lower_bound_minutes are just corrected values when True.
        Defaults to False so existing callers (dashboard, what-if engine)
        get identical behavior to before this arg existed. Only flip the
        default to True after calibrate_rul.check_per_bucket_coverage
        passes for every bucket on the test set.

    Returns dict:
      point_estimate_minutes -- mean across variants (or bias-corrected, if calibrated=True)
      rul_lower_bound_minutes -- min across variants, clipped >= 0
          (or conformal lower bound around the corrected estimate, if calibrated=True)
      std_minutes -- spread across variants (for dashboard confidence display).
          NOT affected by calibrated= -- always the raw ensemble spread.

    This return shape is locked in and will not change across Step A/B/C
    work -- safe to build against now.
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

    if calibrated:
        from calibrate_rul import apply_calibration, load_calibration_params
        y_cal, lb_cal = apply_calibration(point_estimate, load_calibration_params())
        return {
            "point_estimate_minutes": y_cal,
            "rul_lower_bound_minutes": lb_cal,
            "std_minutes": std,
        }

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
    X, y, flight_ids = build_windowed_dataset_v5(flights)
    print(f"Total windows: {len(X)}")

    print("Splitting by flight into train/val/test (not by row) ...")
    unique_flights = np.unique(flight_ids)
    train_flights, temp_flights = train_test_split(unique_flights, test_size=0.3, random_state=42)
    val_flights, test_flights = train_test_split(temp_flights, test_size=0.5, random_state=42)

    train_mask = np.isin(flight_ids, train_flights)
    val_mask = np.isin(flight_ids, val_flights)
    test_mask = np.isin(flight_ids, test_flights)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    train_flight_ids = flight_ids[train_mask]

    print(f"Train windows: {len(X_train)} -- Val windows: {len(X_val)} -- Test windows: {len(X_test)}")
    print(f"Train flights: {len(train_flights)} -- Val flights: {len(val_flights)} -- Test flights: {len(test_flights)}")

    print("Scaling sensor features (fit on train only) ...")
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    X_train_scaled = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    print(f"\nTraining {N_VARIANTS} ensemble variants (bootstrapping + hidden_size + core-aware feature bagging) ...")
    print("NOTE: per-variant RMSE below is on the VALIDATION set. Test stays untouched until the final check.")
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
            X_train_scaled, y_train, train_flight_ids, X_val_scaled, y_val,
            seed, dropout, hidden_size, n_dropped
        )
        dropped_names = [SENSOR_FIELDS[j] for j in dropped_idx]
        print(f"    -> val RMSE: {rmse:.2f}  ({n_params:,} params)  dropped: {dropped_names}")
        models.append(model)
        rmses.append(rmse)
        dropped_idx_list.append(dropped_idx)

    print(f"\nEnsemble val RMSEs: {[round(r, 2) for r in rmses]}")
    print(f"Mean single-model val RMSE: {np.mean(rmses):.2f}  (v3 baseline was 44.94)")

    print("\nSanity check on 5 VALIDATION windows (tuning happens here, test stays untouched):")
    for i in range(min(5, len(X_val))):
        result = predict_rul_ensemble(X_val[i], models, scaler, dropped_idx_list)
        true_val = y_val[i]
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

    # ---------------------------------------------------------------------
    # Step B: fit per-bucket bias correction + conformal calibration on
    # VALIDATION only. This does NOT touch X_test/y_test -- that stays
    # reserved for the one-time coverage check below (run manually, once,
    # after this script finishes -- not part of this __main__ block).
    # ---------------------------------------------------------------------
    print("\nFitting Step B calibration (bias correction + conformal) on VALIDATION set ...")
    from calibrate_rul import fit_calibration
    fit_calibration(X_val, y_val, models, scaler, dropped_idx_list)
    print("\nTest set (X_test_scaled/y_test) is built and scaled above but NOT used yet --")
    print("reserved for the final, one-time coverage check. Run separately:")
    print("  from calibrate_rul import load_calibration_params, check_per_bucket_coverage")
    print("  params = load_calibration_params()")
    print("  check_per_bucket_coverage(X_test_scaled, y_test, models, scaler, dropped_idx_list, params)")
    print("Only flip predict_rul_ensemble's calibrated= default to True after every bucket passes.")

    print("\nTeammates: if you have a raw 7-sensor window, call")
    print("build_inference_window(raw_window) first, then pass the result to")
    print("predict_rul_ensemble(window, models, scaler, dropped_idx_list)")
    print("after loading all variant state_dicts (note: hidden_size differs per variant --")
    print("see HIDDEN_SIZES in this file) and the dropped_idx_list joblib file.")
    print("Pass calibrated=True to get Step B's bias-corrected estimate + conformal lower bound.")