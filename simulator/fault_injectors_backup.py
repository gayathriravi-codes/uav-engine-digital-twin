"""
fault_injectors.py -- all six fault types across the team (roles swapped):
misfire, overheat, cooling_degradation (Aashita);
oil_issue, sensor_drift, vibration_fault (Ashmitha).

Each injector takes a healthy flight DataFrame, picks a random onset point,
perturbs sensors after onset to simulate the fault developing, and returns
a faulted DataFrame with ground-truth RUL columns attached (since the fault
onset/failure point is known by construction).

Run directly to sanity-check + plot: python simulator/fault_injectors.py
"""
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schema import HEALTHY_RANGES
from simulate_healthy import simulate_healthy_flight


def _degradation_curve(linear_progress, convexity=2.5):
    """
    Converts a linear 0->1 progress value into a convex (accelerating) curve,
    calibrated against real turbofan run-to-failure degradation shapes
    (NASA C-MAPSS FD001): sensors stay roughly flat/noisy for the first ~60%
    of remaining life, then decline sharply in the final third, rather than
    degrading in a straight line from onset to failure.

    convexity > 1 => curve stays low longer, then accelerates near the end
    (2.5 is a reasonable match to the FD001 shape observed on sensors
    2, 4, 11, 15, 21 -- flat to ~0.6 life fraction, then convex decline).
    """
    return linear_progress ** convexity


def _add_rul_column(df, onset_idx, failure_idx):
    """
    rul (remaining useful life, in timesteps) counts down to 0 at failure_idx
    for every row from onset_idx onward; before onset, RUL is just the
    distance to failure_idx too (still valid ground truth, just "healthy" RUL).
    """
    n = len(df)
    rul = np.clip(failure_idx - np.arange(n), 0, None).astype(float)
    df["true_rul_timesteps"] = rul
    df["fault_onset_idx"] = onset_idx
    df["failure_idx"] = failure_idx
    return df


def inject_misfire(df, onset_frac=None, severity=1.0, seed=None):
    """
    Misfire: irregular RPM drops (individual cylinder firing failures) with
    corresponding EGT spikes on the affected cycles (unburnt fuel igniting
    late in the exhaust). Gets more frequent/severe as it approaches failure.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)
    onset_idx = int(n * (onset_frac if onset_frac else rng.uniform(0.3, 0.6)))
    failure_idx = n - 1

    df["fault_type"] = "none"
    for i in range(onset_idx, n):
        linear_progress = (i - onset_idx) / max(1, (failure_idx - onset_idx))  # 0 -> 1
        progress = _degradation_curve(linear_progress)  # convex, calibrated to C-MAPSS shape
        misfire_prob = 0.05 + 0.5 * progress * severity
        if rng.random() < misfire_prob:
            df.loc[i, "rpm"] -= rng.uniform(80, 250) * (0.5 + progress)
            df.loc[i, "egt"] += rng.uniform(15, 60) * (0.5 + progress)
            df.loc[i, "vibration"] += rng.uniform(0.3, 1.0) * (0.5 + progress)
            df.loc[i, "fault_type"] = "misfire"

    df = _add_rul_column(df, onset_idx, failure_idx)
    return df


def inject_overheat(df, onset_frac=None, severity=1.0, seed=None):
    """
    Overheating / combustion instability: EGT and CHT both trend upward
    together after onset (distinguish from cooling_degradation, where CHT
    rises but EGT stays flatter -- see Miljkovic's EGT-CHT pattern).
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)
    onset_idx = int(n * (onset_frac if onset_frac else rng.uniform(0.3, 0.6)))
    failure_idx = n - 1

    df["fault_type"] = "none"
    ramp_len = failure_idx - onset_idx
    egt_end_rise = rng.uniform(60, 110) * severity
    cht_end_rise = rng.uniform(40, 80) * severity
    for i in range(onset_idx, n):
        linear_progress = (i - onset_idx) / max(1, ramp_len)
        progress = _degradation_curve(linear_progress)  # convex, calibrated to C-MAPSS shape
        df.loc[i, "egt"] += egt_end_rise * progress + rng.normal(0, 2)
        df.loc[i, "cht"] += cht_end_rise * progress + rng.normal(0, 1.5)
        df.loc[i, "fault_type"] = "overheat"

    df = _add_rul_column(df, onset_idx, failure_idx)
    return df


def inject_cooling_degradation(df, onset_frac=None, severity=1.0, seed=None):
    """
    Cooling degradation: CHT rises steadily while EGT stays relatively flat
    (heat isn't being carried away properly, but combustion itself is fine) --
    the opposite signature from overheat/combustion-instability.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)
    onset_idx = int(n * (onset_frac if onset_frac else rng.uniform(0.3, 0.6)))
    failure_idx = n - 1

    df["fault_type"] = "none"
    ramp_len = failure_idx - onset_idx
    cht_end_rise = rng.uniform(50, 100) * severity
    for i in range(onset_idx, n):
        linear_progress = (i - onset_idx) / max(1, ramp_len)
        progress = _degradation_curve(linear_progress)  # convex, calibrated to C-MAPSS shape
        df.loc[i, "cht"] += cht_end_rise * progress + rng.normal(0, 1.5)
        # EGT gets only a small secondary rise -- this is what separates it from overheat
        df.loc[i, "egt"] += (cht_end_rise * 0.15) * progress + rng.normal(0, 2)
        df.loc[i, "fault_type"] = "cooling_degradation"

    df = _add_rul_column(df, onset_idx, failure_idx)
    return df


def inject_oil_issue(df, onset_frac=None, severity=1.0, seed=None):
    """
    Oil issue: gradual oil pressure drop combined with a slower oil temp rise,
    representing a developing lubrication problem (seal wear, oil breakdown,
    or a slow leak). Other sensors stay normal -- this is a lubrication-side
    fault, not a combustion-side one.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)
    onset_idx = int(n * (onset_frac if onset_frac else rng.uniform(0.3, 0.6)))
    failure_idx = n - 1

    df["fault_type"] = "none"
    ramp_len = failure_idx - onset_idx
    pressure_end_drop = rng.uniform(15, 30) * severity
    temp_end_rise = rng.uniform(20, 40) * severity
    for i in range(onset_idx, n):
        linear_progress = (i - onset_idx) / max(1, ramp_len)
        progress = _degradation_curve(linear_progress)
        df.loc[i, "oil_pressure"] -= pressure_end_drop * progress + rng.normal(0, 0.5)
        df.loc[i, "oil_temp"] += temp_end_rise * progress + rng.normal(0, 1.0)
        df.loc[i, "fault_type"] = "oil_issue"

    df = _add_rul_column(df, onset_idx, failure_idx)
    return df


def inject_sensor_drift(df, onset_frac=None, severity=1.0, seed=None, sub_type=None, target_sensor=None):
    """
    Sensor drift: a SENSOR malfunction, not an engine malfunction -- every
    other sensor keeps behaving normally, which is what makes this fault type
    recognizable as distinct from a real physical degradation.

    Three sub-types (randomly chosen if not specified):
      - "bias"  -- a sudden constant offset appears at onset and stays fixed
      - "drift" -- an offset that grows steadily/linearly over time
      - "stuck" -- the sensor freezes at a constant value regardless of what
                   the engine is actually doing (zero variance after onset)
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)
    onset_idx = int(n * (onset_frac if onset_frac else rng.uniform(0.3, 0.6)))
    failure_idx = n - 1

    sub_type = sub_type if sub_type else rng.choice(["bias", "drift", "stuck"])
    target_sensor = target_sensor if target_sensor else rng.choice(["oil_pressure", "vibration"])

    df["fault_type"] = "none"
    ramp_len = failure_idx - onset_idx
    frozen_value = df.loc[onset_idx, target_sensor] if onset_idx < n else df[target_sensor].iloc[0]
    bias_amount = rng.uniform(0.15, 0.35) * df[target_sensor].mean() * severity

    for i in range(onset_idx, n):
        linear_progress = (i - onset_idx) / max(1, ramp_len)
        if sub_type == "bias":
            df.loc[i, target_sensor] += bias_amount
        elif sub_type == "drift":
            df.loc[i, target_sensor] += bias_amount * linear_progress
        elif sub_type == "stuck":
            df.loc[i, target_sensor] = frozen_value
        df.loc[i, "fault_type"] = "sensor_drift"

    df["drift_sub_type"] = sub_type
    df["drift_target_sensor"] = target_sensor
    df = _add_rul_column(df, onset_idx, failure_idx)
    return df


def inject_vibration_fault(df, onset_frac=None, severity=1.0, seed=None):
    """
    Vibration fault: abnormal vibration signature combining a rising mean
    level (progressive imbalance) with rising variance/spikiness (mounting
    looseness) -- plausible under multiple underlying mechanical causes
    rather than one narrow pattern.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)
    onset_idx = int(n * (onset_frac if onset_frac else rng.uniform(0.3, 0.6)))
    failure_idx = n - 1

    df["fault_type"] = "none"
    ramp_len = failure_idx - onset_idx
    vib_end_rise = rng.uniform(1.0, 2.5) * severity
    for i in range(onset_idx, n):
        linear_progress = (i - onset_idx) / max(1, ramp_len)
        progress = _degradation_curve(linear_progress)
        spike_std = 0.1 + 0.4 * progress
        df.loc[i, "vibration"] += vib_end_rise * progress + rng.normal(0, spike_std)
        df.loc[i, "fault_type"] = "vibration_fault"

    df = _add_rul_column(df, onset_idx, failure_idx)
    return df


INJECTORS = {
    "misfire": inject_misfire,
    "overheat": inject_overheat,
    "cooling_degradation": inject_cooling_degradation,
    "oil_issue": inject_oil_issue,
    "sensor_drift": inject_sensor_drift,
    "vibration_fault": inject_vibration_fault,
}


def generate_fault_dataset(fault_type, n_flights=20, duration=300, out_dir="data/raw"):
    """Generates n_flights labeled flights for one fault type and saves each as a CSV."""
    os.makedirs(out_dir, exist_ok=True)
    injector = INJECTORS[fault_type]
    for i in range(n_flights):
        healthy = simulate_healthy_flight(duration_timesteps=duration, seed=1000 + i)
        faulted = injector(healthy, seed=2000 + i)
        faulted["flight_id"] = f"{fault_type}_{i:03d}"
        path = os.path.join(out_dir, f"{fault_type}_{i:03d}.csv")
        faulted.to_csv(path, index=False)
    print(f"Generated {n_flights} '{fault_type}' flights in {out_dir}/")


def generate_healthy_dataset(n_flights=15, duration=300, out_dir="data/raw"):
    """
    Generates n_flights labeled healthy flights (fault_type='none' throughout).
    Gayatri's classifier needs real 'none' examples to learn from, not just
    the absence of a fault label -- a flight that's healthy start to finish
    looks different from a flight that develops a fault only in its second half.
    """
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n_flights):
        df = simulate_healthy_flight(duration_timesteps=duration, seed=5000 + i)
        df["fault_type"] = "none"
        df["true_rul_timesteps"] = -1.0   # not meaningful for a healthy-only flight
        df["fault_onset_idx"] = -1
        df["failure_idx"] = -1
        df["flight_id"] = f"healthy_{i:03d}"
        path = os.path.join(out_dir, f"healthy_{i:03d}.csv")
        df.to_csv(path, index=False)
    print(f"Generated {n_flights} healthy flights in {out_dir}/")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    healthy = simulate_healthy_flight(duration_timesteps=300, seed=42)

    fig, axes = plt.subplots(6, 2, figsize=(12, 20))
    plot_sensors = {
        "misfire": ["egt", "cht"],
        "overheat": ["egt", "cht"],
        "cooling_degradation": ["egt", "cht"],
        "oil_issue": ["oil_pressure", "oil_temp"],
        "sensor_drift": ["oil_pressure", "vibration"],
        "vibration_fault": ["vibration"],
    }
    for row, (name, fn) in enumerate(INJECTORS.items()):
        faulted = fn(healthy, onset_frac=0.4, seed=7)
        for sensor in plot_sensors[name]:
            axes[row, 0].plot(faulted["timestamp"], faulted[sensor], label=sensor.upper())
        axes[row, 0].axvline(faulted["fault_onset_idx"].iloc[0], color="red", linestyle="--", label="onset")
        sensor_label = "/".join(s.upper() for s in plot_sensors[name])
        axes[row, 0].set_title(f"{name} -- {sensor_label}")
        axes[row, 0].legend(fontsize=7)
        axes[row, 1].plot(faulted["timestamp"], faulted["true_rul_timesteps"])
        axes[row, 1].set_title(f"{name} -- true RUL (ground truth)")

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sanity_check_plot_v2.png")
    try:
        plt.savefig(plot_path, dpi=100)
        print(f"Saved {plot_path} -- open it and confirm curves look physically plausible")
    except OSError as e:
        print(f"Could not save plot (file may be open in another program): {e}")
        print("Continuing to generate the dataset anyway -- the plot is just a visual check.")

    # generate the real dataset (18 flights per fault type = ~108 total across 6 fault types)
    for ft in INJECTORS:
        generate_fault_dataset(ft, n_flights=18, out_dir="data/raw")

    # generate healthy-only flights (needed as a real "none" class for the classifier)
    generate_healthy_dataset(n_flights=15, out_dir="data/raw")