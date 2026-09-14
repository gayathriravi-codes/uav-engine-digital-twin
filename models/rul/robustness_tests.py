"""
robustness_tests.py -- model robustness testing for the RUL ensemble.
Ashmitha's hackathon-day task: stress-test predict_rul_ensemble() against
sensor noise, core-sensor dropout, short flights, and multi-fault windows.

Run: python models/rul/robustness_tests.py   (from project root, AeroTwin/)
"""
import sys, os
sys.path.insert(0, 'models/rul')
import numpy as np

from train_rul_ensemble import (
    load_ensemble, load_all_flights, build_inference_window,
    build_windowed_dataset_v5, WINDOW_SIZE, _compute_derived_features,
)
from train_rul_ensemble import predict_rul_ensemble
from schema import SENSOR_FIELDS


def test_1_sensor_noise(models, scaler, dropped_idx_list, sample_window, sample_label):
    print("\n=== TEST 1: Sensor Noise Sensitivity ===")
    print(f"True RUL for this window: {sample_label:.1f}")

    baseline = predict_rul_ensemble(sample_window, models, scaler, dropped_idx_list)
    print(f"Baseline (no noise): point_est={baseline['point_estimate_timesteps']:.1f}  "
          f"lower_bound={baseline['rul_lower_bound_timesteps']:.1f}  std={baseline['std_timesteps']:.1f}")

    feature_names = ["egt_slope", "integrated_deviation", "vibration_variance"]
    n_steps = sample_window.shape[0]

    for noise_pct in [0.01, 0.05, 0.10]:
        noise_scale = noise_pct * np.abs(sample_window[:, :7]).mean()
        print(f"\n  --- Noise level: {noise_pct:.0%} ---")

        # ---------------- Variant A: independent (i.i.d.) noise ----------------
        # Original harsh test -- independent Gaussian noise per timestep,
        # per sensor. Worst-case, not typical-case: real sensor noise is
        # usually smoother/correlated between adjacent readings.
        rng_a = np.random.RandomState(42)
        noisy_window_a = sample_window.copy()
        noise_a = rng_a.normal(0, noise_scale, noisy_window_a[:, :7].shape)
        noisy_window_a[:, :7] += noise_a

        new_derived_a = _compute_derived_features(noisy_window_a[:, :7])
        noisy_window_a[:, 7:] = np.tile(new_derived_a, (n_steps, 1))

        print(f"    Variant A (independent noise):")
        for name, val in zip(feature_names, new_derived_a):
            print(f"      [debug] {name} = {val:.4f}")
        print(f"      [debug] any NaN/Inf in derived features? {not np.all(np.isfinite(new_derived_a))}")

        result_a = predict_rul_ensemble(noisy_window_a, models, scaler, dropped_idx_list)
        shift_a = abs(result_a['point_estimate_timesteps'] - baseline['point_estimate_timesteps'])
        print(f"      point_est={result_a['point_estimate_timesteps']:.1f}  "
              f"(shift={shift_a:.1f})  std={result_a['std_timesteps']:.1f}")

        # ---------------- Variant B: smoothed/correlated noise ----------------
        # Same noise magnitude, but run through a cumulative running average
        # so adjacent noise values are correlated instead of independent --
        # closer to how real sensor noise/drift typically behaves.
        rng_b = np.random.RandomState(42)
        noisy_window_b = sample_window.copy()
        raw_noise_b = rng_b.normal(0, noise_scale, noisy_window_b[:, :7].shape)
        smoothed_noise_b = np.cumsum(raw_noise_b, axis=0) / np.arange(1, n_steps + 1)[:, None]
        noisy_window_b[:, :7] += smoothed_noise_b

        new_derived_b = _compute_derived_features(noisy_window_b[:, :7])
        noisy_window_b[:, 7:] = np.tile(new_derived_b, (n_steps, 1))

        print(f"    Variant B (smoothed/correlated noise):")
        for name, val in zip(feature_names, new_derived_b):
            print(f"      [debug] {name} = {val:.4f}")
        print(f"      [debug] any NaN/Inf in derived features? {not np.all(np.isfinite(new_derived_b))}")

        result_b = predict_rul_ensemble(noisy_window_b, models, scaler, dropped_idx_list)
        shift_b = abs(result_b['point_estimate_timesteps'] - baseline['point_estimate_timesteps'])
        print(f"      point_est={result_b['point_estimate_timesteps']:.1f}  "
              f"(shift={shift_b:.1f})  std={result_b['std_timesteps']:.1f}")


def test_2_core_sensor_dropout(models, scaler, dropped_idx_list, sample_window, sample_label):
    print("\n=== TEST 2: Core Sensor Dropout ===")
    print(f"True RUL for this window: {sample_label:.1f}")

    baseline = predict_rul_ensemble(sample_window, models, scaler, dropped_idx_list)
    print(f"Baseline (all sensors): point_est={baseline['point_estimate_timesteps']:.1f}  std={baseline['std_timesteps']:.1f}")

    RUL_CORE_SENSORS = ['vibration', 'cht', 'oil_pressure', 'rpm']
    for sensor in RUL_CORE_SENSORS:
        idx = SENSOR_FIELDS.index(sensor)
        dropped_window = sample_window.copy()
        dropped_window[:, idx] = 0.0

        # Recompute derived features from the zeroed raw sensor, so any
        # feature depending on it reflects the dropout instead of holding
        # a stale pre-dropout value.
        new_derived = _compute_derived_features(dropped_window[:, :7])
        dropped_window[:, 7:] = np.tile(new_derived, (dropped_window.shape[0], 1))

        result = predict_rul_ensemble(dropped_window, models, scaler, dropped_idx_list)
        shift = abs(result['point_estimate_timesteps'] - baseline['point_estimate_timesteps'])
        print(f"  {sensor} zeroed: point_est={result['point_estimate_timesteps']:.1f}  "
              f"(shift={shift:.1f})  std={result['std_timesteps']:.1f}")


def test_3_short_flight(models, scaler, dropped_idx_list, flights):
    print("\n=== TEST 3: Short Flight Edge Case ===")
    # Find the shortest flight that still has at least one full window
    shortest = min(flights, key=lambda f: len(f[1]))
    flight_id, df = shortest
    print(f"Shortest flight found: {flight_id}, length={len(df)} timesteps")

    if len(df) < WINDOW_SIZE:
        print(f"  Flight is SHORTER than window_size={WINDOW_SIZE} -- no valid window can be built.")
        print(f"  This confirms create_windows_v5 correctly skips flights below window_size (see range() bound).")
        return

    sensor_data = df[SENSOR_FIELDS].values
    raw_window = sensor_data[0:WINDOW_SIZE]  # first available window
    # build_inference_window computes derived features internally from this
    # real, untouched slice of flight data, so it is already consistent --
    # no post-hoc recompute needed here (unlike tests 1/2/4, which modify
    # raw sensors in place after the window is already built).
    full_window = build_inference_window(raw_window)
    true_rul = df["true_rul_timesteps"].values[WINDOW_SIZE - 1]

    result = predict_rul_ensemble(full_window, models, scaler, dropped_idx_list)
    print(f"  True RUL: {true_rul:.1f}  point_est={result['point_estimate_timesteps']:.1f}  "
          f"lower_bound={result['rul_lower_bound_timesteps']:.1f}  std={result['std_timesteps']:.1f}")
    print(f"  Sanity: lower_bound <= point_est? {result['rul_lower_bound_timesteps'] <= result['point_estimate_timesteps']}")
    print(f"  Sanity: lower_bound >= 0? {result['rul_lower_bound_timesteps'] >= 0}")


def test_4_multi_fault_window(models, scaler, dropped_idx_list, sample_window):
    print("\n=== TEST 4: Simulated Multi-Fault Signature (out-of-distribution input) ===")
    baseline = predict_rul_ensemble(sample_window, models, scaler, dropped_idx_list)
    print(f"Baseline (single real fault pattern): std={baseline['std_timesteps']:.1f}")

    # Combine an oil-issue-like signature (oil_pressure down, oil_temp up) with a
    # vibration-fault-like signature (vibration up) in the SAME window -- a pattern
    # the model has never seen combined during training (each fault type is
    # injected independently, never two at once).
    combo_window = sample_window.copy()
    oil_pressure_idx = SENSOR_FIELDS.index('oil_pressure')
    oil_temp_idx = SENSOR_FIELDS.index('oil_temp')
    vibration_idx = SENSOR_FIELDS.index('vibration')

    combo_window[:, oil_pressure_idx] -= 20
    combo_window[:, oil_temp_idx] += 25
    combo_window[:, vibration_idx] += 1.5

    # Recompute derived features from the modified raw sensors -- same fix
    # as tests 1/2, so the multi-fault window is internally consistent
    # rather than mixing modified raw sensors with stale derived features.
    new_derived = _compute_derived_features(combo_window[:, :7])
    combo_window[:, 7:] = np.tile(new_derived, (combo_window.shape[0], 1))

    result = predict_rul_ensemble(combo_window, models, scaler, dropped_idx_list)
    print(f"Combined oil_issue + vibration_fault signature: point_est={result['point_estimate_timesteps']:.1f}  "
          f"std={result['std_timesteps']:.1f}")
    print(f"  Did ensemble std INCREASE for this out-of-distribution combo? "
          f"{'YES (good -- model is appropriately less certain)' if result['std_timesteps'] > baseline['std_timesteps'] else 'NO (flat -- may be overconfident on unfamiliar patterns)'}")


if __name__ == "__main__":
    print("Loading ensemble ...")
    models, scaler, dropped_idx_list = load_ensemble()

    print("Loading flights and building a sample window ...")
    flights = load_all_flights()
    X, y, flight_ids = build_windowed_dataset_v5(flights)

    # Pick a representative mid-RUL sample window for tests 1, 2, 4
    mid_idx = np.argmin(np.abs(y - 100))  # closest window to true RUL=100
    sample_window = X[mid_idx]
    sample_label = y[mid_idx]

    test_1_sensor_noise(models, scaler, dropped_idx_list, sample_window, sample_label)
    test_2_core_sensor_dropout(models, scaler, dropped_idx_list, sample_window, sample_label)
    test_3_short_flight(models, scaler, dropped_idx_list, flights)
    test_4_multi_fault_window(models, scaler, dropped_idx_list, sample_window)

    print("\n=== ROBUSTNESS TEST SUITE COMPLETE ===")