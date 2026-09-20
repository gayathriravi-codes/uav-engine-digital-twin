"""
mission_hysteresis.py
-----------------------
Thin stateful wrapper around recommend_mission_action() that adds
hysteresis: downgrading to a LESS severe action requires the raw
recommendation to agree for `downgrade_streak` consecutive ticks,
while escalating to a MORE severe action is always applied immediately.

Why: raw per-tick recommendations can genuinely flap near a threshold
(e.g. ABORT -> RETURN_TO_BASE -> ABORT within a few ticks as smoothed
RUL oscillates right around the margin ratio) - found via demo_playback.py
testing on overheat_000.csv. This mirrors how real avionics/telemetry
systems avoid reacting to single-sample noise, while never delaying a
genuine escalation to a more severe state.
"""

from mission_reliability import MissionAction, recommend_mission_action

# Severity ranking - lower index = more severe
SEVERITY = [
    MissionAction.ABORT,
    MissionAction.RETURN_TO_BASE,
    MissionAction.REDUCE_LOAD,
    MissionAction.CONTINUE,
]


class HysteresisGate:
    def __init__(self, downgrade_streak=2):
        """
        downgrade_streak: number of consecutive ticks the raw recommendation
        must agree on a LESS severe action before we actually downgrade.
        Escalating to MORE severe is never delayed - only recovery is.
        """
        self.downgrade_streak = downgrade_streak
        self.last_action = None
        self._pending_action = None
        self._pending_count = 0

    def update(self, **kwargs):
        rec = recommend_mission_action(**kwargs)
        raw_action = rec.action

        if self.last_action is None:
            self.last_action = raw_action
            return rec

        raw_rank = SEVERITY.index(raw_action)
        last_rank = SEVERITY.index(self.last_action)

        if raw_rank <= last_rank:
            # same or more severe - apply immediately, no delay
            self.last_action = raw_action
            self._pending_action = None
            self._pending_count = 0
            return rec

        # raw_action is LESS severe than what we're currently showing -
        # require it to persist before downgrading
        if raw_action == self._pending_action:
            self._pending_count += 1
        else:
            self._pending_action = raw_action
            self._pending_count = 1

        if self._pending_count >= self.downgrade_streak:
            self.last_action = raw_action
            self._pending_count = 0
            self._pending_action = None
            return rec

        # not enough consecutive agreement yet - hold at last_action
        rec.action = self.last_action
        rec.reasons = [
            f"raw recommendation is {raw_action.value}, holding at "
            f"{self.last_action.value} pending confirmation "
            f"({self._pending_count}/{self.downgrade_streak})"
        ] + rec.reasons
        return rec