"""
battery_injection_sensors.py -- battery/alternator health, injection timing,
and injector-abnormality modeling for PS 26054 Section B/C gap closure.

DELIBERATELY SEPARATE from schema.py's SENSOR_FIELDS and simulate_healthy.py.
The 108+123-flight main dataset and every trained model (RUL ensemble,
XGBoost classifier) are built on exactly the 7 sensors in SENSOR_FIELDS.
Adding these as first-class sensors there would force a full retrain of
everything and risk the leak-fixed 91.1% classifier result -- same reasoning
that kept RECOVERY_INJECTORS out of the main INJECTORS dict in
fault_injectors.py.

This module generates its own small, additive dataset: two new sensor
columns (battery_voltage, injection_timing_deg) attached on top of existing
healthy/faulted flights, plus three new fault injectors. Demoable and
traceability-matrix-worthy on their own, with zero risk to existing models.

Run directly to sanity-check + generate a demo dataset:
    python simulator/battery_injection_sensors.py
"""
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulate_healthy import simulate_healthy_flight


# --- Healthy ranges for the two new sensors (not added to schema.py's
# HEALTHY_RANGES -- these are standalone, same reasoning as above) ---
NEW_SENSOR_HEALTHY_RANGES = {
    "battery_voltage": (26.0, 29.0),      # typical 28V aircraft DC bus, healthy alternator charging
    "injection_timing_deg": (18.0, 22.0),  # degrees BTDC, nominal injection advance
}


def _degradation_curve(linear_progress, convexity=2.5):
    """Same convex shape as fault_injectors.py, kept local so this module
    has no hidden coupling to that file beyond the healthy simulator."""
    return linear_progress ** convexity


def attach_healthy_battery_injection_columns(df, seed=None):
    """
    Adds battery_voltage and injection_timing_deg columns to an existing
    flight DataFrame (healthy or already-faulted), populated with plausible
    healthy noise around the nominal values. Call this on every flight
    before applying any of this module's fault injectors below.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)

    # Battery voltage: alternator charging holds it near the top of the
    # healthy range with small ripple; slow random walk keeps it realistic
    # rather than pure white noise.
    v_lo, v_hi = NEW_SENSOR_HEALTHY_RANGES["battery_voltage"]
    v_center = (v_lo + v_hi) / 2
    battery_walk = np.cumsum(rng.normal(0, 0.03, n))
    battery_walk -= battery_walk.mean()  # keep it centered, no net drift
    df["battery_voltage"] = v_center + battery_walk + rng.normal(0, 0.05, n)

    # Injection timing: mechanically governed, very stable in a healthy
    # engine -- small noise only, no walk.
    t_lo, t_hi = NEW_SENSOR_HEALTHY_RANGES["injection_timing_deg"]
    t_center = (t_lo + t_hi) / 2
    df["injection_timing_deg"] = t_center + rng.normal(0, 0.3, n)

    return df


def _onset_and_failure(n, onset_frac, rng):
    onset_idx = int(n * (onset_frac if onset_frac else rng.uniform(0.3, 0.6)))
    failure_idx = n - 1
    return onset_idx, failure_idx


def _add_rul_column(df, onset_idx, failure_idx):
    """Same convention as fault_injectors.py's _add_rul_column."""
    n = len(df)
    rul = np.clip(failure_idx - np.arange(n), 0, None).astype(float)
    df["true_rul_timesteps"] = rul
    df["fault_onset_idx"] = onset_idx
    df["failure_idx"] = failure_idx
    return df


def inject_battery_alternator_fault(df, onset_frac=None, severity=1.0, seed=None):
    """
    Battery/alternator health fault: a failing alternator can't keep the
    bus voltage up under load, so battery_voltage sags steadily after onset,
    AND the ripple/variance around that sagging mean increases (a degrading
    alternator's rectification gets noisier, not just lower on average --
    this is what separates "alternator failing" from "battery just a bit
    low"). Every other sensor, including the 7 core ones, is untouched:
    this is an electrical-system fault, not an engine-combustion fault.

    Requires df to already have battery_voltage (call
    attach_healthy_battery_injection_columns first).
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)
    onset_idx, failure_idx = _onset_and_failure(n, onset_frac, rng)
    ramp_len = failure_idx - onset_idx

    df["fault_type"] = "none"
    voltage_end_drop = rng.uniform(3.0, 6.0) * severity  # e.g. 28V -> ~22-25V near failure
    for i in range(onset_idx, n):
        linear_progress = (i - onset_idx) / max(1, ramp_len)
        progress = _degradation_curve(linear_progress)
        ripple_std = 0.05 + 0.35 * progress  # noisier as alternator degrades
        df.loc[i, "battery_voltage"] -= voltage_end_drop * progress
        df.loc[i, "battery_voltage"] += rng.normal(0, ripple_std)
        df.loc[i, "fault_type"] = "battery_alternator_fault"

    df = _add_rul_column(df, onset_idx, failure_idx)
    return df


def inject_injection_timing_fault(df, onset_frac=None, severity=1.0, seed=None, direction=None):
    """
    Injection timing fault: timing drifts away from nominal (either advanced
    or retarded -- randomly chosen if not specified) after onset. Retarded
    timing tends to raise EGT slightly (combustion finishing later, more
    heat exhausted rather than doing work); advanced timing can raise
    cylinder pressure/vibration slightly. Both effects are kept SMALL and
    secondary here -- injection_timing_deg itself is the primary, clearly
    abnormal signal, not a re-hash of overheat/vibration_fault.

    Requires df to already have injection_timing_deg AND the 7 core sensors
    (egt, vibration) present, since it applies a small secondary effect to
    those.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)
    onset_idx, failure_idx = _onset_and_failure(n, onset_frac, rng)
    ramp_len = failure_idx - onset_idx

    direction = direction if direction else rng.choice(["advanced", "retarded"])
    timing_end_shift = rng.uniform(4.0, 8.0) * severity
    if direction == "retarded":
        timing_end_shift = -timing_end_shift

    df["fault_type"] = "none"
    for i in range(onset_idx, n):
        linear_progress = (i - onset_idx) / max(1, ramp_len)
        progress = _degradation_curve(linear_progress)
        df.loc[i, "injection_timing_deg"] += timing_end_shift * progress + rng.normal(0, 0.3)
        if direction == "retarded" and "egt" in df.columns:
            df.loc[i, "egt"] += 15 * severity * progress  # small secondary effect only
        elif direction == "advanced" and "vibration" in df.columns:
            df.loc[i, "vibration"] += 0.3 * severity * progress  # small secondary effect only
        df.loc[i, "fault_type"] = "injection_timing_fault"

    df["timing_fault_direction"] = direction
    df = _add_rul_column(df, onset_idx, failure_idx)
    return df


def inject_injector_abnormality(df, onset_frac=None, severity=1.0, seed=None):
    """
    Injector abnormality (fuel injector clogging/leaking/sticking), kept
    DISTINCT from misfire on purpose:

      - misfire's signature: sharp, discrete RPM drops + EGT spikes on the
        affected cycles.
      - injector abnormality's signature here: fuel_flow becomes irregular
        (pulsing / inconsistent delivery -- mean drifts down as if
        partially clogged, AND variance rises as if intermittently
        sticking/leaking), while RPM stays essentially normal and EGT only
        rises mildly and smoothly (leaner mixture from reduced fuel, not
        the sharp spike pattern of misfire).

    This is what should let a classifier tell the two apart rather than
    this fault type collapsing back into "misfire under another name."
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)
    onset_idx, failure_idx = _onset_and_failure(n, onset_frac, rng)
    ramp_len = failure_idx - onset_idx

    df["fault_type"] = "none"
    flow_end_drop = rng.uniform(2.0, 5.0) * severity
    for i in range(onset_idx, n):
        linear_progress = (i - onset_idx) / max(1, ramp_len)
        progress = _degradation_curve(linear_progress)
        pulse_std = 0.3 + 1.2 * progress  # rising irregularity, not a sharp spike
        df.loc[i, "fuel_flow"] -= flow_end_drop * progress
        df.loc[i, "fuel_flow"] += rng.normal(0, pulse_std)
        df.loc[i, "egt"] += 8 * severity * progress  # mild, smooth rise -- not misfire's spike pattern
        df.loc[i, "fault_type"] = "injector_abnormality"

    df = _add_rul_column(df, onset_idx, failure_idx)
    return df


NEW_INJECTORS = {
    "battery_alternator_fault": inject_battery_alternator_fault,
    "injection_timing_fault": inject_injection_timing_fault,
    "injector_abnormality": inject_injector_abnormality,
}


def generate_new_fault_dataset(fault_type, n_flights=15, duration=300, out_dir="data/raw_new_sensors", seed_base=3000):
    """
    Generates a small, separate demo dataset for one of the three new fault
    types -- kept in data/raw_new_sensors/, not data/raw/, so the existing
    108+123-flight dataset and all trained models remain untouched.
    """
    os.makedirs(out_dir, exist_ok=True)
    injector = NEW_INJECTORS[fault_type]
    for i in range(n_flights):
        healthy = simulate_healthy_flight(duration_timesteps=duration, seed=seed_base + i)
        healthy = attach_healthy_battery_injection_columns(healthy, seed=seed_base + i + 500)
        faulted = injector(healthy, seed=seed_base + i + 1000)
        faulted["flight_id"] = f"{fault_type}_{i:03d}"
        path = os.path.join(out_dir, f"{fault_type}_{i:03d}.csv")
        faulted.to_csv(path, index=False)
    print(f"Generated {n_flights} '{fault_type}' flights in {out_dir}/ (separate from main dataset)")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    healthy = simulate_healthy_flight(duration_timesteps=300, seed=42)
    healthy = attach_healthy_battery_injection_columns(healthy, seed=42)

    print("Healthy battery_voltage range:", healthy["battery_voltage"].min(), "-", healthy["battery_voltage"].max())
    print("Healthy injection_timing_deg range:", healthy["injection_timing_deg"].min(), "-", healthy["injection_timing_deg"].max())

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    plot_sensors = {
        "battery_alternator_fault": ["battery_voltage"],
        "injection_timing_fault": ["injection_timing_deg", "egt"],
        "injector_abnormality": ["fuel_flow", "egt"],
    }
    for row, (name, fn) in enumerate(NEW_INJECTORS.items()):
        faulted = fn(healthy, onset_frac=0.4, seed=7)
        for sensor in plot_sensors[name]:
            axes[row, 0].plot(faulted["timestamp"], faulted[sensor], label=sensor.upper())
        axes[row, 0].axvline(faulted["fault_onset_idx"].iloc[0], color="red", linestyle="--", label="onset")
        axes[row, 0].set_title(f"{name}")
        axes[row, 0].legend(fontsize=7)
        axes[row, 1].plot(faulted["timestamp"], faulted["true_rul_timesteps"])
        axes[row, 1].set_title(f"{name} -- true RUL (ground truth)")

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "new_sensors_sanity_check.png")
    try:
        plt.savefig(plot_path, dpi=100)
        print(f"Saved {plot_path} -- open it and confirm curves look physically plausible")
    except OSError as e:
        print(f"Could not save plot (file may be open in another program): {e}")

    for ft in NEW_INJECTORS:
        generate_new_fault_dataset(ft, n_flights=15, out_dir="data/raw_new_sensors")