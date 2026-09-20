"""
mission_reliability.py
-----------------------
Mission-reliability decision layer for AeroTwin.

Turns engine-level predictions (RUL, health score, fault type/severity)
into a mission-level recommendation: CONTINUE / REDUCE_LOAD / RTB / ABORT.

This directly closes the gap between the PS title ("...Mission Reliability
Enhancement...") and the current engine-only prediction stack. It is a
rules layer, not a new ML model, so it's feasible to build and test before
the deadline, and it's easy to explain/defend in Q&A because every
threshold is inspectable and tunable live.

Integration point: call recommend_mission_action() once per inference
cycle, after you already have:
  - rul_estimate_timesteps / rul_lower_bound_timesteps (from predict_rul_ensemble)
  - health_score (from calculate_health_score)
  - fault_type, fault_confidence (from predict_fault)
  - mission_remaining_timesteps (planner/GCS provides this - if you don't
    have real mission planning data, simulate it for the demo)
"""

from dataclasses import dataclass, field
from enum import Enum


class MissionAction(str, Enum):
    CONTINUE = "CONTINUE"
    REDUCE_LOAD = "REDUCE_LOAD"
    RETURN_TO_BASE = "RETURN_TO_BASE"
    ABORT = "ABORT"


# Faults where "reduce load" is a physically meaningful mitigation.
# Faults NOT in this set skip straight to RTB/ABORT logic if severe,
# since reducing load doesn't help (e.g. sensor_drift doesn't respond
# to load changes - matches your own diagnosis of that fault type).
LOAD_RESPONSIVE_FAULTS = {"oil_issue", "vibration_fault", "overheat"}

# Tunable thresholds - surface these as sliders on the dashboard so
# judges can see the decision logic is transparent, not a black box.
THRESHOLDS = {
    "abort_rul_margin_ratio": 1.1,   # RUL_lower_bound < mission_remaining * this -> ABORT
    "rtb_rul_margin_ratio": 1.5,     # RUL_lower_bound < mission_remaining * this -> RTB
    "reduce_load_health_score": 70,  # health_score below this -> consider load reduction
    "abort_health_score": 30,        # health_score below this -> ABORT regardless of RUL
    "min_fault_confidence": 0.6,     # ignore low-confidence fault classifications
}


@dataclass
class MissionRecommendation:
    action: MissionAction
    risk_score: float          # 0-100, higher = worse
    reasons: list = field(default_factory=list)
    rul_margin_timesteps: float = None


def recommend_mission_action(
    rul_estimate_timesteps: float,
    rul_lower_bound_timesteps: float,
    health_score: float,
    fault_type: str,
    fault_confidence: float,
    mission_remaining_timesteps: float,
    thresholds: dict = THRESHOLDS,
) -> MissionRecommendation:
    reasons = []
    margin = rul_lower_bound_timesteps - mission_remaining_timesteps

    # --- No active fault: RUL model has no real countdown-to-failure signal
    # to extrapolate from on genuinely healthy windows (true_rul_timesteps
    # is a sentinel/unset value for healthy training data), so its RUL
    # output here is not a meaningful failure estimate. Skip RUL-margin
    # gates entirely and rely on health_score alone for this case. ---
    if fault_type == "none":
        if health_score < thresholds["abort_health_score"]:
            reasons.append(
                f"no active fault, but health_score {health_score:.1f} "
                f"below abort threshold"
            )
            return MissionRecommendation(
                MissionAction.ABORT, risk_score=95.0, reasons=reasons,
                rul_margin_timesteps=None,
            )
        reasons.append(
            "no active fault detected, health_score nominal - "
            "RUL margin check skipped (not meaningful without an "
            "active fault trajectory to extrapolate from)"
        )
        return MissionRecommendation(
            MissionAction.CONTINUE, risk_score=5.0, reasons=reasons,
            rul_margin_timesteps=None,
        )

    # --- Hard safety gate: health score crisis overrides everything ---
    if health_score < thresholds["abort_health_score"]:
        reasons.append(
            f"health_score {health_score:.1f} below abort threshold "
            f"{thresholds['abort_health_score']}"
        )
        return MissionRecommendation(
            MissionAction.ABORT, risk_score=95.0, reasons=reasons,
            rul_margin_timesteps=margin,
        )

    # --- Absolute floor: RUL exhausted is ABORT regardless of mission_remaining.
    # (Without this, mission_remaining=0 makes the ratio checks below degenerate
    # to "0 < 0", which is False, silently disabling both ABORT and RTB gates
    # right when they matter most - found via demo_playback.py testing at t=300.) ---
    if rul_lower_bound_timesteps <= 0:
        reasons.append(
            f"RUL lower bound exhausted ({rul_lower_bound_timesteps:.0f}) "
            f"regardless of mission remaining"
        )
        return MissionRecommendation(
            MissionAction.ABORT, risk_score=99.0, reasons=reasons,
            rul_margin_timesteps=margin,
        )

    # --- RUL-vs-mission-remaining safety margin check ---
    if rul_lower_bound_timesteps < mission_remaining_timesteps * thresholds["abort_rul_margin_ratio"]:
        reasons.append(
            f"RUL lower bound ({rul_lower_bound_timesteps:.0f}) leaves "
            f"insufficient margin over remaining mission "
            f"({mission_remaining_timesteps:.0f})"
        )
        return MissionRecommendation(
            MissionAction.ABORT, risk_score=90.0, reasons=reasons,
            rul_margin_timesteps=margin,
        )

    if rul_lower_bound_timesteps < mission_remaining_timesteps * thresholds["rtb_rul_margin_ratio"]:
        reasons.append(
            f"RUL lower bound margin is thin relative to mission "
            f"remaining - recommend early return"
        )
        return MissionRecommendation(
            MissionAction.RETURN_TO_BASE, risk_score=65.0, reasons=reasons,
            rul_margin_timesteps=margin,
        )

    # --- Degraded but survivable: try load reduction if the fault responds to it ---
    if (
        health_score < thresholds["reduce_load_health_score"]
        and fault_confidence >= thresholds["min_fault_confidence"]
        and fault_type in LOAD_RESPONSIVE_FAULTS
    ):
        reasons.append(
            f"health_score {health_score:.1f} degraded, fault_type "
            f"'{fault_type}' is load-responsive -> recommend load reduction"
        )
        return MissionRecommendation(
            MissionAction.REDUCE_LOAD,
            risk_score=45.0,
            reasons=reasons,
            rul_margin_timesteps=margin,
        )

    # --- Degraded, but fault type doesn't respond to load reduction ---
    # (e.g. sensor_drift: reducing load doesn't fix an unreliable reading -
    # matches your existing diagnosis that sensor_drift needs a different
    # response, not a physical mitigation)
    if health_score < thresholds["reduce_load_health_score"]:
        reasons.append(
            f"health_score {health_score:.1f} degraded but fault_type "
            f"'{fault_type}' does not respond to load reduction - "
            f"monitor closely, no physical mitigation available"
        )
        return MissionRecommendation(
            MissionAction.CONTINUE,
            risk_score=55.0,
            reasons=reasons,
            rul_margin_timesteps=margin,
        )

    reasons.append("all metrics within nominal bounds")
    return MissionRecommendation(
        MissionAction.CONTINUE, risk_score=10.0, reasons=reasons,
        rul_margin_timesteps=margin,
    )


if __name__ == "__main__":
    # Quick smoke test with made-up numbers - replace with real pipeline output
    rec = recommend_mission_action(
        rul_estimate_timesteps=180,
        rul_lower_bound_timesteps=150,
        health_score=62,
        fault_type="oil_issue",
        fault_confidence=0.82,
        mission_remaining_timesteps=90,
    )
    print(rec)