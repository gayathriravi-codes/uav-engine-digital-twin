"""
diagnose_collapse.py -- one-off check: is the -0.0 collapse coming from
the raw ensemble mean going negative, or from calibration?
Run: python models\rul\diagnose_collapse.py   (from project root, AeroTwin/)
"""
import sys, os
sys.path.insert(0, 'models/rul')
import numpy as np

from train_rul_ensemble import (
    load_ensemble, load_all_flights, build_windowed_dataset_v5,
    _compute_derived_features, predict_rul_ensemble,
)

print("Loading ensemble ...")
models, scaler, dropped_idx_list = load_ensemble()

print("Loading flights and rebuilding the same sample window as robustness_tests.py ...")
flights = load_all_flights()
X, y, flight_ids = build_windowed_dataset_v5(flights)
mid_idx = np.argmin(np.abs(y - 100))
sample_window = X[mid_idx]
sample_label = y[mid_idx]
n_steps = sample_window.shape[0]

# Reproduce Variant B (smoothed/correlated) noise at 1%, same seed, same
# construction as test_1_sensor_noise.
noise_pct = 0.01
noise_scale = noise_pct * np.abs(sample_window[:, :7]).mean()
rng_b = np.random.RandomState(42)
noisy_window_b = sample_window.copy()
raw_noise_b = rng_b.normal(0, noise_scale, noisy_window_b[:, :7].shape)
smoothed_noise_b = np.cumsum(raw_noise_b, axis=0) / np.arange(1, n_steps + 1)[:, None]
noisy_window_b[:, :7] += smoothed_noise_b
new_derived_b = _compute_derived_features(noisy_window_b[:, :7])
noisy_window_b[:, 7:] = np.tile(new_derived_b, (n_steps, 1))

print(f"\nTrue RUL: {sample_label:.1f}")
print(f"integrated_deviation for this noisy window: {new_derived_b[1]:.2f}")

result_raw = predict_rul_ensemble(noisy_window_b, models, scaler, dropped_idx_list, calibrated=False)
result_cal = predict_rul_ensemble(noisy_window_b, models, scaler, dropped_idx_list, calibrated=True)

print(f"\ncalibrated=False (raw ensemble mean/min/std):")
print(f"  point_est={result_raw['point_estimate_timesteps']:.4f}  "
      f"lower_bound={result_raw['rul_lower_bound_timesteps']:.4f}  std={result_raw['std_timesteps']:.4f}")

print(f"\ncalibrated=True (Step B bias-corrected + conformal):")
print(f"  point_est={result_cal['point_estimate_timesteps']:.4f}  "
      f"lower_bound={result_cal['rul_lower_bound_timesteps']:.4f}  std={result_cal['std_timesteps']:.4f}")

if result_raw['point_estimate_timesteps'] <= 0.001:
    print("\n=> Raw ensemble mean is ALREADY at/near zero before calibration.")
    print("   The feature fix (mean instead of sum) is the relevant lever here.")
else:
    print("\n=> Raw ensemble mean is NOT near zero -- calibration is producing the -0.0.")
    print("   Look at calibrate_rul.py's per-bucket correction/clip logic instead of the feature.")