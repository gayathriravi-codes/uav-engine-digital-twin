"""
human_override.py
--------------------
Human-override framing: decides whether a mission recommendation should
auto-execute or require operator confirmation before acting, based on
the confidence of the fault classification behind it.

Why: a rules-based mission-reliability layer is transparent and
defensible (see mission_reliability.py's own docstring), but "transparent"
isn't the same as "always correct" - a high-severity action (ABORT/
RETURN_TO_BASE) triggered on a LOW-confidence fault classification is
exactly the case where a human operator should confirm before the system
acts unilaterally, not after. Low-severity actions (CONTINUE/REDUCE_LOAD)
auto-execute regardless of confidence, since the cost of a wrong
low-severity action is much smaller than the cost of a wrong high-severity
one - this mirrors real aviation automation design (autopilot escalations
typically require more confirmation than routine adjustments).
"""

from mission_reliability import MissionAction

# Actions in this set require operator confirmation if fault_confidence
# is below CONFIDENCE_THRESHOLD. Actions NOT in this set always
# auto-execute, since a wrong low-severity action is cheap to correct.
HIGH_STAKES_ACTIONS = {MissionAction.ABORT, MissionAction.RETURN_TO_BASE}

# Below this confidence, a high-stakes action requires operator
# confirmation rather than auto-executing. Tuned against real confidence
# distribution across all 123 flights (check_confidence_range.py):
# global minimum confidence was 0.75 (overheat_012.csv), and several
# flights dip into 0.81-0.87 at fault onset - 0.85 is set to actually
# trigger on real, validated low-confidence moments rather than sit
# dormant at an arbitrary round number that never fires in practice.
CONFIDENCE_THRESHOLD = 0.85

def requires_operator_confirmation(action, fault_confidence):
    """
    Returns True if this action should be held for operator confirmation
    rather than auto-executed.
    """
    if action not in HIGH_STAKES_ACTIONS:
        return False
    return fault_confidence < CONFIDENCE_THRESHOLD


def annotate_recommendation(rec, fault_confidence):
    """
    Takes a MissionRecommendation and the fault_confidence it was based
    on, and returns (requires_confirmation: bool, note: str or None).
    Does not modify rec - the caller decides what to do with the flag
    (e.g. hold auto-execution, surface a confirmation prompt, log it).
    """
    needs_confirmation = requires_operator_confirmation(
        rec.action, fault_confidence
    )

    if needs_confirmation:
        note = (
            f"{rec.action.value} recommended on fault_confidence="
            f"{fault_confidence:.2f}, below the {CONFIDENCE_THRESHOLD:.2f} "
            f"threshold for auto-execution - requires operator confirmation "
            f"before acting."
        )
    else:
        note = None

    return needs_confirmation, note