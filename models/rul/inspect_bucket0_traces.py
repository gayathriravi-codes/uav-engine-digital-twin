"""
inspect_bucket0_traces.py

Pulls raw sensor traces for the failing bucket-(0,50) VALIDATION windows
and compares them against healthy bucket-(0,50) validation windows, to
check whether any raw channel (not the existing derived features) separates
them. Mirrors the exact split logic in run_coverage_check.py / train_rul_ensemble.py.

Val-only. Does not touch test. Safe to re-run and edit freely.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rul import load_all_flights
from train_rul_ensemble import build_windowed_dataset_v5
from schema import SENSOR_FIELDS

# Known failing TEST-set window indices from diagnose_bucket0.py / prior sessions.
# NOTE: these were indices into X_test, not X_val -- see the check below.
KNOWN_FAILING_TEST_IDX = list(range(655, 660)) + list(range(710, 715)) + list(range(760, 770))

N_HEALTHY_COMPARISON = 6


def main():
    flights = load_all_flights()
    X, y, flight_ids = build_windowed_dataset_v5(flights)

    # Exact same split as run_coverage_check.py / train_rul_ensemble.py
    unique_flights = np.unique(flight_ids)
    train_flights, temp_flights = train_test_split(unique_flights, test_size=0.3, random_state=42)
    val_flights, test_flights = train_test_split(temp_flights, test_size=0.5, random_state=42)

    val_mask = np.isin(flight_ids, val_flights)
    test_mask = np.isin(flight_ids, test_flights)

    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"Val windows: {len(X_val)}  Test windows: {len(X_test)}")

    # IMPORTANT: the known failing indices (655-659, 710-714, 760-769) came from
    # diagnose_bucket0.py run against X_test -- they are positions in X_test,
    # NOT X_val. The current val-side failure (13/160 flagged, 89.4% coverage)
    # is a DIFFERENT set of windows on a different split. Find the actual
    # low-RUL-with-high-error windows freshly, on val, rather than reusing
    # test indices that don't apply here.

    bucket0_val_mask = (y_val >= 0) & (y_val < 50)
    bucket0_idx = np.where(bucket0_val_mask)[0]
    print(f"Bucket (0,50) val windows: {len(bucket0_idx)}")

    # Need models/scaler to know which of these are actually failing coverage
    # on val -- reuse load_ensemble + calibration exactly as check_or_gate_sweep did.
    from train_rul_ensemble import load_ensemble, predict_rul_ensemble
    from calibrate_rul import load_calibration_params

    models, scaler, dropped_idx_list = load_ensemble()
    params = load_calibration_params()

    failing_val_idx = []
    healthy_val_idx = []
    for i in bucket0_idx:
        result = predict_rul_ensemble(X_val[i], models, scaler, dropped_idx_list, calibrated=True)
        true_rul = y_val[i]
        covered = result["rul_lower_bound_timesteps"] <= true_rul
        if not covered:
            failing_val_idx.append(i)
        elif len(healthy_val_idx) < N_HEALTHY_COMPARISON:
            healthy_val_idx.append(i)

    print(f"Failing (not covered) val windows in bucket (0,50): {len(failing_val_idx)} -> {failing_val_idx}")
    print(f"Healthy comparison windows: {healthy_val_idx}")

    if not failing_val_idx:
        print("\nNo failing windows found on val for bucket (0,50) -- coverage may already be >=90%")
        print("on this exact call path, or the failure is test-specific. Re-check against")
        print("check_or_gate_sweep.py's per-window logic if this seems inconsistent.")
        return

    # --- Numeric comparison: last-10-sample slope per raw channel ---
    print("\n=== Per-channel stats: failing vs healthy (raw sensors only) ===")
    for ch_idx, ch_name in enumerate(SENSOR_FIELDS):
        fail_traces = [X_val[i][:, ch_idx] for i in failing_val_idx]
        healthy_traces = [X_val[i][:, ch_idx] for i in healthy_val_idx]

        fail_slope = [np.polyfit(range(len(t[-10:])), t[-10:], 1)[0] for t in fail_traces]
        healthy_slope = [np.polyfit(range(len(t[-10:])), t[-10:], 1)[0] for t in healthy_traces]

        print(f"\n{ch_name}:")
        print(f"  failing  last-10 slope: mean={np.mean(fail_slope):.4f}  std={np.std(fail_slope):.4f}")
        print(f"  healthy  last-10 slope: mean={np.mean(healthy_slope):.4f}  std={np.std(healthy_slope):.4f}")

    # --- Also compare true_rul vs point_estimate directly for the failing set ---
    print("\n=== Failing windows: true_rul vs point_estimate vs lower_bound ===")
    for i in failing_val_idx:
        result = predict_rul_ensemble(X_val[i], models, scaler, dropped_idx_list, calibrated=True)
        print(f"  idx={i:4d}  true={y_val[i]:6.1f}  point_est={result['point_estimate_timesteps']:7.1f}  "
              f"lb={result['rul_lower_bound_timesteps']:6.1f}")

    # --- Visual comparison ---
    fig, axes = plt.subplots(len(SENSOR_FIELDS), 1, figsize=(10, 3 * len(SENSOR_FIELDS)), sharex=True)
    for ch_idx, (ax, ch_name) in enumerate(zip(axes, SENSOR_FIELDS)):
        for i in failing_val_idx:
            ax.plot(X_val[i][:, ch_idx], color="red", alpha=0.6,
                    label="failing" if i == failing_val_idx[0] else None)
        for i in healthy_val_idx:
            ax.plot(X_val[i][:, ch_idx], color="gray", alpha=0.4,
                    label="healthy" if i == healthy_val_idx[0] else None)
        ax.set_title(ch_name)
        ax.legend()

    plt.tight_layout()
    out_path = "bucket0_trace_comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot -> {out_path}")


if __name__ == "__main__":
    main()