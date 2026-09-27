"""
combustion_instability_injector.py -- closes the last open item in PS 26054
Section C (Fault Detection): combustion instability, previously only
loosely proxied by misfire.

Designed to be genuinely separable from all three existing "similar" faults,
not a relabeling of any of them:

  - misfire: SHARP, DISCRETE RPM drops + EGT spikes on individual cycles.
  - vibration_fault: rising MEAN + variance on vibration specifically.
  - injector_abnormality: fuel_flow pulsing/decline, mild EGT rise, RPM
    left essentially untouched.
  - combustion_instability (this file): small-amplitude, CONTINUOUS
    cycle-to-cycle JITTER on RPM and EGT together (correlated -- both
    fluctuate erratically in the same pattern, since irregular combustion
    directly affects both crank speed and exhaust temperature every
    cycle), with NO discrete drops/spikes, NO fuel_flow involvement, and
    NO vibration involvement. The signature is rising *irregularity*
    (coefficient of variation), not a mean shift in either sensor.

Writes its own demo dataset into data/raw_new_sensors/ -- the SAME
directory battery_injection_sensors.py's new fault types use, so
feature_extraction_v2.py picks this up automatically on the next run
with no changes needed there.

Run from repo root: python simulator/combustion_instability_injector.py
"""
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulate_healthy import simulate_healthy_flight
from battery_injection_sensors import attach_healthy_battery_injection_columns


def _degradation_curve(linear_progress, convexity=2.5):
    """Same convex shape used throughout the project (calibrated against
    C-MAPSS run-to-failure data) -- kept local, same choice
    battery_injection_sensors.py made, to avoid cross-file coupling."""
    return linear_progress ** convexity


def _onset_and_failure(n, onset_frac, rng):
    onset_idx = int(n * (onset_frac if onset_frac else rng.uniform(0.3, 0.6)))
    failure_idx = n - 1
    return onset_idx, failure_idx


def _add_rul_column(df, onset_idx, failure_idx):
    n = len(df)
    rul = np.clip(failure_idx - np.arange(n), 0, None).astype(float)
    df["true_rul_timesteps"] = rul
    df["fault_onset_idx"] = onset_idx
    df["failure_idx"] = failure_idx
    return df


def inject_combustion_instability(df, onset_frac=None, severity=1.0, seed=None):
    """
    Combustion instability: after onset, injects CORRELATED cycle-to-cycle
    jitter on rpm and egt (both driven by the same underlying erratic-firing
    signal, scaled to each sensor's own range) with rising amplitude, but
    no directional mean shift and no discrete spike/drop events. This is
    what separates it from misfire (discrete, uncorrelated-in-onset-timing
    drops), overheat (monotonic mean rise), and vibration_fault (rising
    mean+variance on vibration only).

    fuel_flow and vibration are left completely untouched, so a classifier
    should be able to tell this apart from injector_abnormality (which
    touches fuel_flow) and vibration_fault (which touches vibration).
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)
    onset_idx, failure_idx = _onset_and_failure(n, onset_frac, rng)
    ramp_len = failure_idx - onset_idx

    df["fault_type"] = "none"
    rpm_jitter_end_std = rng.uniform(30, 70) * severity   # rpm healthy range is ~4800-5200
    egt_jitter_end_std = rng.uniform(8, 20) * severity    # egt healthy range is ~680-760

    for i in range(onset_idx, n):
        linear_progress = (i - onset_idx) / max(1, ramp_len)
        progress = _degradation_curve(linear_progress)

        # Shared underlying "erratic firing" signal, scaled differently per
        # sensor -- this is what makes rpm and egt jitter CORRELATED rather
        # than two independent noise sources, matching the real physical
        # link (irregular combustion affects both at once).
        shared_signal = rng.normal(0, 1.0)
        df.loc[i, "rpm"] += shared_signal * rpm_jitter_end_std * progress
        df.loc[i, "egt"] += shared_signal * egt_jitter_end_std * progress
        # small independent component too, so it's not a perfect 1:1 copy
        df.loc[i, "rpm"] += rng.normal(0, 5 * progress)
        df.loc[i, "egt"] += rng.normal(0, 2 * progress)

        df.loc[i, "fault_type"] = "combustion_instability"

    df = _add_rul_column(df, onset_idx, failure_idx)
    return df


def generate_combustion_instability_dataset(n_flights=15, duration=300,
                                             out_dir="data/raw_new_sensors", seed_base=6000):
    """
    Generates a small demo dataset, written into the SAME directory as
    battery_injection_sensors.py's new fault types, so a single
    feature_extraction_v2.py run picks up all 4 new fault types together.
    """
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n_flights):
        healthy = simulate_healthy_flight(duration_timesteps=duration, seed=seed_base + i)
        healthy = attach_healthy_battery_injection_columns(healthy, seed=seed_base + i + 500)
        faulted = inject_combustion_instability(healthy, seed=seed_base + i + 1000)
        faulted["flight_id"] = f"combustion_instability_{i:03d}"
        path = os.path.join(out_dir, f"combustion_instability_{i:03d}.csv")
        faulted.to_csv(path, index=False)
    print(f"Generated {n_flights} 'combustion_instability' flights in {out_dir}/")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    healthy = simulate_healthy_flight(duration_timesteps=300, seed=42)
    healthy = attach_healthy_battery_injection_columns(healthy, seed=42)
    faulted = inject_combustion_instability(healthy, onset_frac=0.4, seed=7)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    axes[0].plot(faulted["timestamp"], faulted["rpm"], label="RPM", color="tab:blue")
    axes[0].axvline(faulted["fault_onset_idx"].iloc[0], color="red", linestyle="--", label="onset")
    axes[0].set_title("combustion_instability -- RPM (should show rising jitter, no mean shift)")
    axes[0].legend(fontsize=8)

    axes[1].plot(faulted["timestamp"], faulted["egt"], label="EGT", color="tab:orange")
    axes[1].axvline(faulted["fault_onset_idx"].iloc[0], color="red", linestyle="--", label="onset")
    axes[1].set_title("combustion_instability -- EGT (should jitter in step with RPM above)")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "combustion_instability_sanity_check.png")
    try:
        plt.savefig(plot_path, dpi=100)
        print(f"Saved {plot_path} -- open it and confirm RPM/EGT jitter TOGETHER, with no mean shift, no discrete drops/spikes")
    except OSError as e:
        print(f"Could not save plot (file may be open in another program): {e}")

    generate_combustion_instability_dataset(n_flights=15)