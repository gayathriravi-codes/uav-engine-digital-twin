"""
demo_playback.py
------------------
Live-playback demo script: steps through a flight, decrementing
mission_remaining_timesteps as the "mission clock" counts down,
while engine state is inferred fresh at each step and smoothed
via MissionSmoother.

This is the actual demo script to run live at judging - it prints
a running narrative of engine state + mission clock + recommendation
at each step, so you can walk judges through the escalation as it
happens rather than showing a static table.

Designed around overheat_000.csv, which showed the cleanest full
CONTINUE -> RETURN_TO_BASE -> ABORT escalation during testing -
but works with any flight file.

Also writes an append-only JSONL decision/audit log (one JSON object
per tick) to AUDIT_LOG_PATH, capturing everything the mission-reliability
layer decided and why - a near-free byproduct since MissionRecommendation
already produces this content. Each entry also records whether that
tick's action requires operator confirmation before auto-executing
(see human_override.py).
"""
import json
from datetime import datetime, timezone
import sys
import time
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.classification.engine_inference import run_engine_inference
from mission_reliability import recommend_mission_action
from mission_smoothing import MissionSmoother
from mission_hysteresis import HysteresisGate
from human_override import annotate_recommendation

# --- Demo configuration ---
FLIGHT_PATH = "data/raw/overheat_000.csv"
START_ROW = 150            # earliest point we have enough rows (>=30) to infer from
STEP = 5                   # rows advanced per "tick" - smaller = smoother countdown
MISSION_CLOCK_START = 150  # mission_remaining_timesteps at demo start
PAUSE_SECONDS = 0.0        # set >0 (e.g. 0.5) for a slower, more watchable live demo
AUDIT_LOG_PATH = "logs/decision_audit_log.jsonl"


def run_playback(flight_path=FLIGHT_PATH, start_row=START_ROW, step=STEP,
                  mission_clock_start=MISSION_CLOCK_START, pause=PAUSE_SECONDS):
    full_df = pd.read_csv(flight_path)
    smoother = MissionSmoother(window=3)
    gate = HysteresisGate(downgrade_streak=2)
    mission_clock = mission_clock_start
    last_action = None

    Path(AUDIT_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    log_file = open(AUDIT_LOG_PATH, "a", encoding="utf-8")
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print("=" * 78)
    print(f"AEROTWIN LIVE MISSION PLAYBACK - {Path(flight_path).name}")
    print("=" * 78)

    for cutoff in range(start_row, len(full_df) + 1, step):
        df = full_df.iloc[:cutoff].copy()
        engine_state = run_engine_inference(df)
        smoothed = smoother.update(engine_state)

        rec = gate.update(
            rul_estimate_timesteps=smoothed["rul_estimate_timesteps"],
            rul_lower_bound_timesteps=smoothed["rul_lower_bound_timesteps"],
            health_score=smoothed["health_score"],
            fault_type=smoothed["predicted_fault"],
            fault_confidence=smoothed["confidence"],
            mission_remaining_timesteps=mission_clock,
        )

        requires_confirmation, override_note = annotate_recommendation(
            rec, smoothed["confidence"]
        )

        changed_marker = "  <<< ACTION CHANGED" if rec.action.value != last_action else ""
        confirmation_marker = "  [REQUIRES OPERATOR CONFIRMATION]" if requires_confirmation else ""
        print(f"[t={cutoff:3}] mission_clock={mission_clock:3} | "
              f"fault={smoothed['predicted_fault']:6} conf={smoothed['confidence']:.2f} | "
              f"health={smoothed['health_score']:6} | "
              f"RUL_lb(smoothed)={smoothed['rul_lower_bound_timesteps']:6} | "
              f"-> {rec.action.value:16}{changed_marker}{confirmation_marker}")

        if rec.action.value != last_action and last_action is not None:
            print(f"         reason: {rec.reasons[0]}")

        if override_note:
            print(f"         override: {override_note}")

        log_entry = {
            "session_id": session_id,
            "flight_file": Path(flight_path).name,
            "tick": cutoff,
            "mission_clock": mission_clock,
            "predicted_fault": smoothed["predicted_fault"],
            "fault_confidence": smoothed["confidence"],
            "health_score": smoothed["health_score"],
            "rul_estimate_timesteps": smoothed["rul_estimate_timesteps"],
            "rul_lower_bound_timesteps": smoothed["rul_lower_bound_timesteps"],
            "action": rec.action.value,
            "risk_score": rec.risk_score,
            "reasons": rec.reasons,
            "action_changed": rec.action.value != last_action,
            "requires_operator_confirmation": requires_confirmation,
            "override_note": override_note,
        }
        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()

        last_action = rec.action.value
        mission_clock = max(0, mission_clock - step)  # mission clock counts down each tick

        if pause > 0:
            time.sleep(pause)

    log_file.close()
    print(f"Decision audit log written to: {AUDIT_LOG_PATH}")
    print("=" * 78)
    print("PLAYBACK COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    run_playback()