"""
distribution_shift_check.py

Sanity check for the mismatch sweep: does a 15% engine-build coupling
spread actually make individual engines look different from each other?
If it does, and macro F1 still stays flat, the robustness result is real.
If it doesn't, the perturbation was too weak to prove anything.

Run from the project root:
    python distribution_shift_check.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "classification"))

from mismatch_sweep import make_builds, generate_flights_for_build
from schema import SENSOR_FIELDS

SPREADS = [0.0, 0.05, 0.15]
N_UNITS = 6
N_HEALTHY = 10
N_PER_FAULT = 1   # generator is called with this; faulty flights are filtered out below


def healthy_only(df: pd.DataFrame) -> pd.DataFrame:
    if "fault_type" in df.columns:
        return df[df["fault_type"] == "none"]
    return df


def per_build_sensor_means(spread: float) -> pd.DataFrame:
    """Rows = builds, columns = sensors, values = mean over healthy flights."""
    rows = []
    for build in make_builds(spread, N_UNITS):
        flights = generate_flights_for_build(build, N_HEALTHY, N_PER_FAULT)
        frames = [healthy_only(f) for f in flights]
        frames = [f for f in frames if len(f) > 0]
        data = pd.concat(frames, ignore_index=True)
        rows.append({s: data[s].mean() for s in SENSOR_FIELDS if s in data.columns})
    return pd.DataFrame(rows)


def main():
    results = {}
    for spread in SPREADS:
        print(f"Generating builds at {spread:.0%} spread...")
        results[spread] = per_build_sensor_means(spread)

    sensors = list(results[SPREADS[0]].columns)
    grand_mean = results[0.0].mean()

    table = pd.DataFrame(index=sensors)
    for spread in SPREADS:
        # between-build spread of the per-build mean, as % of the nominal mean
        table[f"between-build std @ {spread:.0%} (% of mean)"] = (
            results[spread].std() / grand_mean.abs() * 100
        )

    base_col = f"between-build std @ {SPREADS[0]:.0%} (% of mean)"
    top_col = f"between-build std @ {SPREADS[-1]:.0%} (% of mean)"
    table["ratio 15% / 0%"] = table[top_col] / table[base_col].replace(0, np.nan)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print("\nHow different do individual engine builds look from each other?")
    print("(per-build mean of each sensor on healthy flights)\n")
    print(table.to_string())

    print("\nHow to read this:")
    print("- At 0% spread every build is identical, so the number is just")
    print("  flight-to-flight noise (the floor).")
    print("- If the 15% number is clearly above the 0% floor for sensors that")
    print("  have an rpm coupling (egt, cht, oil_pressure, oil_temp, vibration,")
    print("  fuel_flow), the perturbation is real and the flat F1 curve is")
    print("  a genuine robustness result.")
    print("- rpm has no coupling constant, so no shift there is expected.")
    print("- If ALL sensors sit at the floor even at 15%, the perturbation is")
    print("  too weak (or not wired in) and the sweep proves nothing yet.")


if __name__ == "__main__":
    main()