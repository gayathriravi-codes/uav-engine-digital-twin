"""
distribution_shift_check_v2.py

Measures what coupling-constant spread actually changes: the slope of each
sensor against rpm (and each sensor's std), per engine build, on healthy
flights. Compares between-build variation at 0% vs 5% vs 15% spread.

Run from the project root:
    python distribution_shift_check_v2.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "classification"))

from mismatch_sweep import make_builds, generate_flights_for_build

SPREADS = [0.0, 0.05, 0.15]
N_UNITS = 6
N_HEALTHY = 10
N_PER_FAULT = 1
COUPLED = ["egt", "cht", "oil_pressure", "oil_temp", "vibration", "fuel_flow"]


def healthy_only(df):
    if "fault_type" in df.columns:
        return df[df["fault_type"] == "none"]
    return df


def per_build_stats(spread):
    """Returns (slopes, stds): rows = builds, columns = sensors."""
    slope_rows, std_rows = [], []
    for build in make_builds(spread, N_UNITS):
        flights = generate_flights_for_build(build, N_HEALTHY, N_PER_FAULT)
        frames = [healthy_only(f) for f in flights]
        frames = [f for f in frames if len(f) > 1]
        data = pd.concat(frames, ignore_index=True)
        slope_rows.append({s: np.polyfit(data["rpm"], data[s], 1)[0] for s in COUPLED})
        std_rows.append({s: data[s].std() for s in COUPLED})
    return pd.DataFrame(slope_rows), pd.DataFrame(std_rows)


def main():
    slopes, stds = {}, {}
    for spread in SPREADS:
        print(f"Generating builds at {spread:.0%} spread...")
        slopes[spread], stds[spread] = per_build_stats(spread)

    ref_slope = slopes[0.0].mean().abs()
    ref_std = stds[0.0].mean()

    table = pd.DataFrame(index=COUPLED)
    for spread in SPREADS:
        table[f"slope between-build std @ {spread:.0%} (% of nominal)"] = (
            slopes[spread].std() / ref_slope * 100
        )
    table["slope ratio 15%/0%"] = (
        table[f"slope between-build std @ {SPREADS[-1]:.0%} (% of nominal)"]
        / table[f"slope between-build std @ {SPREADS[0]:.0%} (% of nominal)"]
    )
    for spread in SPREADS:
        table[f"sensor std between-build @ {spread:.0%} (% of nominal)"] = (
            stds[spread].std() / ref_std * 100
        )

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda v: f"{v:.2f}")
    print("\nBuild-to-build variation in sensor-vs-rpm slope and sensor std")
    print(table.to_string())
    print("\nSlope ratio well above 1 (say 3x or more) = perturbation is real.")
    print("Ratio near 1 = perturbation is not reaching the generated data.")


if __name__ == "__main__":
    main()