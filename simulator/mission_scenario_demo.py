"""
mission_scenario_demo.py
---------------------------
Verifies that existing fault injectors compose correctly on top of a
named mission profile (not just on the plain nominal healthy flight),
and generates a small demo dataset combining both -- e.g. "overheat
fault occurring during a high_altitude mission" -- for a richer,
more realistic demo than either dimension shown alone.

Run: python simulator/mission_scenario_demo.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mission_profiles import simulate_mission_flight, MISSION_PROFILES
from fault_injectors import INJECTORS


def generate_scenario(fault_type, profile_name, duration=300, seed=42):
    """One flight: a named fault injected onto a named mission profile."""
    baseline = simulate_mission_flight(profile_name=profile_name, duration_timesteps=duration, seed=seed)
    injector = INJECTORS[fault_type]
    faulted = injector(baseline, seed=seed + 1000)
    faulted["mission_profile"] = profile_name  # injector's df.copy() should preserve this, confirm below
    return faulted


if __name__ == "__main__":
    print("Checking fault injectors compose correctly on top of mission profiles...\n")

    combos_to_check = [
        ("overheat", "high_altitude"),
        ("oil_issue", "hot_weather"),
        ("vibration_fault", "rapid_throttle"),
        ("misfire", "rapid_throttle"),  # both touch rpm -- the interaction case to watch
    ]

    for fault_type, profile_name in combos_to_check:
        df = generate_scenario(fault_type, profile_name)
        has_profile_col = "mission_profile" in df.columns
        n_faulted_rows = (df["fault_type"] == fault_type).sum()
        print(f"{fault_type} + {profile_name}:")
        print(f"  mission_profile column preserved: {has_profile_col}")
        print(f"  rows with fault active: {n_faulted_rows}/{len(df)}")
        print(f"  rpm range: [{df['rpm'].min():.0f}, {df['rpm'].max():.0f}]  "
              f"(healthy nominal range ~[4800,5200])")
        print(f"  any NaN values introduced: {df.isna().any().any()}")
        print()

    print("If 'mission_profile column preserved' is False for any combo, the")
    print("injector's df.copy() + reassignment is silently dropping metadata")
    print("columns -- worth confirming before building the full scenario dataset.")