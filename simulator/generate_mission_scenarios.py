"""
generate_mission_scenarios.py
--------------------------------
Generates demo flights for each mission profile (nominal, high_altitude,
endurance, hot_weather, rapid_throttle), optionally with a fault injected,
and saves them into data/raw/ so they appear in the dashboard's existing
flight picker (get_flights() in dashboard/app.py) with no dashboard
changes needed.

Run: python simulator/generate_mission_scenarios.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mission_profiles import simulate_mission_flight, MISSION_PROFILES
from fault_injectors import INJECTORS

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")


def save_scenario(df, name):
    path = os.path.join(OUT_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"Saved {path}  ({len(df)} rows)")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    # Healthy version of each mission profile -- shows the environmental
    # effect alone, no fault, matching REQUIRED_COLUMNS shape as a plain
    # healthy flight (fault_type='none' throughout, like healthy_*.csv).
    for profile_name in MISSION_PROFILES:
        if profile_name == "nominal":
            continue  # nominal is identical to existing healthy_*.csv files, skip duplicating
        df = simulate_mission_flight(profile_name=profile_name, duration_timesteps=300, seed=100)
        df["fault_type"] = "none"
        df["true_rul_timesteps"] = -1.0
        df["fault_onset_idx"] = -1
        df["failure_idx"] = -1
        df["flight_id"] = f"mission_{profile_name}_healthy"
        save_scenario(df, f"mission_{profile_name}_healthy")

    # A few meaningful fault+profile combos for a richer demo narrative --
    # e.g. "overheat develops faster/worse under high_altitude conditions"
    # is a genuinely interesting, defensible story to tell live.
    demo_combos = [
        ("overheat", "high_altitude"),
        ("oil_issue", "hot_weather"),
        ("vibration_fault", "rapid_throttle"),
        ("misfire", "endurance"),
    ]
    for fault_type, profile_name in demo_combos:
        baseline = simulate_mission_flight(profile_name=profile_name, duration_timesteps=300, seed=200)
        faulted = INJECTORS[fault_type](baseline, seed=300)
        faulted["flight_id"] = f"mission_{profile_name}_{fault_type}"
        save_scenario(faulted, f"mission_{profile_name}_{fault_type}")

    print("\nDone. These should now appear in the dashboard's flight picker")
    print("(sidebar dropdown, populated by get_flights() globbing data/raw/*.csv).")