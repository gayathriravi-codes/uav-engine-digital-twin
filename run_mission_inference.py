"""
run_mission_inference.py
--------------------------
Integration script: chains engine_inference.py's run_engine_inference()
output straight into mission_reliability.py's recommend_mission_action().

This is the "real values" version of the smoke test - same function,
but every input now comes from your actual trained models running on
a real flight, instead of hand-typed placeholder numbers.

mission_remaining_timesteps has no real source yet (no mission planner
exists in the project) - for the demo, it's passed in manually below.
Two demo options are shown: a fixed value, and one derived from the
flight's own true_rul_timesteps for a controlled "what-if" scenario.
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.classification.engine_inference import run_engine_inference
from mission_reliability import recommend_mission_action


def run_mission_pipeline(csv_path, mission_remaining_timesteps):
    df = pd.read_csv(csv_path)

    engine_state = run_engine_inference(df)

    recommendation = recommend_mission_action(
        rul_estimate_timesteps=engine_state["rul_estimate_timesteps"],
        rul_lower_bound_timesteps=engine_state["rul_lower_bound_timesteps"],
        health_score=engine_state["health_score"],
        fault_type=engine_state["predicted_fault"],
        fault_confidence=engine_state["confidence"],
        mission_remaining_timesteps=mission_remaining_timesteps,
    )

    return engine_state, recommendation


if __name__ == "__main__":
    csv_path = PROJECT_ROOT / "data" / "raw" / "overheat_000.csv"

    # --- Demo option A: fixed mission-remaining value ---
    MISSION_REMAINING_FIXED = 100

    print("=" * 70)
    print(f"MISSION-RELIABILITY INFERENCE - {csv_path.name}")
    print(f"(mission_remaining_timesteps = {MISSION_REMAINING_FIXED}, fixed)")
    print("=" * 70)

    engine_state, recommendation = run_mission_pipeline(
        csv_path, MISSION_REMAINING_FIXED
    )

    print("\nENGINE STATE:")
    for k, v in engine_state.items():
        print(f"  {k:30}: {v}")

    print("\nMISSION RECOMMENDATION:")
    print(f"  action            : {recommendation.action}")
    print(f"  risk_score        : {recommendation.risk_score}")
    print(f"  rul_margin        : {recommendation.rul_margin_timesteps}")
    print(f"  reasons           : {recommendation.reasons}")
    print("=" * 70)