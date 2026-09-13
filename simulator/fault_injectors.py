"""
fault_injectors.py — Aashita's three fault types (swapped from Ashmitha):
misfire, overheat, cooling_degradation.

Each injector takes a healthy flight DataFrame, picks a random onset point,
perturbs sensors after onset to simulate the fault developing, and returns
(faulted_df, true_rul_series) where true_rul_series[i] = timesteps remaining
until "failure" as measured from row i (ground truth, since WE inject it).

Run directly to sanity-check + plot: python simulator/fault_injectors.py
"""
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


INJECTORS = {
    "misfire": inject_misfire,
    "overheat": inject_overheat,
    "cooling_degradation": inject_cooling_degradation,
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


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    healthy = simulate_healthy_flight(duration_timesteps=300, seed=42)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for row, (name, fn) in enumerate(INJECTORS.items()):
        faulted = fn(healthy, onset_frac=0.4, seed=7)
        axes[row, 0].plot(faulted["timestamp"], faulted["egt"], label="EGT")
        axes[row, 0].plot(faulted["timestamp"], faulted["cht"], label="CHT")
        axes[row, 0].axvline(faulted["fault_onset_idx"].iloc[0], color="red", linestyle="--", label="onset")
        axes[row, 0].set_title(f"{name} — EGT/CHT")
        axes[row, 0].legend(fontsize=7)

        axes[row, 1].plot(faulted["timestamp"], faulted["true_rul_timesteps"])
        axes[row, 1].set_title(f"{name} — true RUL (ground truth)")

    plt.tight_layout()
    plt.savefig("simulator/sanity_check_plot.png", dpi=100)
    print("Saved sanity_check_plot.png -- open it and confirm curves look physically plausible")

    # quick smoke test of the dataset generator (small batch)
    for ft in INJECTORS:
        generate_fault_dataset(ft, n_flights=2, out_dir="data/raw")