"""
mission_smoothing.py
----------------------
Rolling-average smoothing layer between engine_inference's per-window
output and mission_reliability's recommend_mission_action().

Why: confirmed via check_rul_cutoff.py that rul_lower_bound_timesteps
is non-monotonic window-to-window even during genuine degradation
(e.g. misfire_000: 122 -> 146 -> 91 -> 46 -> 46 -> 60 across consecutive
10-row cutoffs) - this is real point-estimate noise in the RUL ensemble,
not a mission_reliability.py bug. Acting on a single noisy reading risks
a spurious RTB/ABORT flip-flop; smoothing over the last few readings
gives a steadier, more defensible signal for mission-level decisions -
consistent with how real telemetry/avionics systems avoid reacting to
a single raw sample.

Also majority-votes the fault_type classification over the same window,
so a single flickered tick (e.g. "none" for one tick in the middle of
a sustained misfire) doesn't reset mission state - found via
demo_playback.py testing on misfire_000.csv, where a lone "none" tick
dropped an active ABORT straight to CONTINUE.

Usage: keep one MissionSmoother per active flight/session (it holds
state across calls). Call .update(engine_state) once per inference
cycle; it returns a smoothed copy of engine_state with rul_estimate_timesteps,
rul_lower_bound_timesteps, and health_score replaced by their rolling
averages, and predicted_fault/confidence replaced by a majority vote
and matching averaged confidence. Pass THAT into recommend_mission_action(),
not the raw engine_state.
"""

from collections import deque


class MissionSmoother:
    def __init__(self, window=3):
        """
        window: number of recent readings to average over.
        3 is a reasonable starting point - large enough to damp
        single-window noise, small enough to still react within a
        few inference cycles to a real, sustained trend.
        """
        self.window = window
        self.rul_estimate_hist = deque(maxlen=window)
        self.rul_lb_hist = deque(maxlen=window)
        self.health_hist = deque(maxlen=window)
        self.fault_type_hist = deque(maxlen=window)
        self.fault_confidence_hist = deque(maxlen=window)

    def update(self, engine_state: dict) -> dict:
        self.rul_estimate_hist.append(engine_state["rul_estimate_timesteps"])
        self.rul_lb_hist.append(engine_state["rul_lower_bound_timesteps"])
        self.health_hist.append(engine_state["health_score"])
        self.fault_type_hist.append(engine_state["predicted_fault"])
        self.fault_confidence_hist.append(engine_state["confidence"])

        smoothed = dict(engine_state)  # shallow copy, keep all other keys unchanged
        smoothed["rul_estimate_timesteps"] = round(
            sum(self.rul_estimate_hist) / len(self.rul_estimate_hist), 2
        )
        smoothed["rul_lower_bound_timesteps"] = round(
            sum(self.rul_lb_hist) / len(self.rul_lb_hist), 2
        )
        smoothed["health_score"] = round(
            sum(self.health_hist) / len(self.health_hist), 2
        )

        # Majority-vote fault type over the window, instead of trusting a
        # single tick's classification.
        most_common_fault = max(
            set(self.fault_type_hist),
            key=self.fault_type_hist.count,
        )
        smoothed["predicted_fault"] = most_common_fault
        # Average confidence only over ticks that agreed with the winning
        # vote, so a flickered tick doesn't drag confidence down either.
        agreeing_confidences = [
            conf for fault, conf in zip(self.fault_type_hist, self.fault_confidence_hist)
            if fault == most_common_fault
        ]
        smoothed["confidence"] = round(
            sum(agreeing_confidences) / len(agreeing_confidences), 2
        )

        smoothed["_smoothing_n_samples"] = len(self.rul_lb_hist)  # useful for debug/demo display
        return smoothed


if __name__ == "__main__":
    # Quick smoke test with the actual noisy misfire sequence we found
    smoother = MissionSmoother(window=3)
    fake_readings = [
        {"rul_estimate_timesteps": 122.26, "rul_lower_bound_timesteps": 122.26, "health_score": 100, "predicted_fault": "misfire", "confidence": 0.9},
        {"rul_estimate_timesteps": 145.76, "rul_lower_bound_timesteps": 145.76, "health_score": 100, "predicted_fault": "misfire", "confidence": 0.92},
        {"rul_estimate_timesteps": 91.01,  "rul_lower_bound_timesteps": 91.01,  "health_score": 100, "predicted_fault": "none",    "confidence": 0.99},
        {"rul_estimate_timesteps": 45.82,  "rul_lower_bound_timesteps": 45.82,  "health_score": 96,  "predicted_fault": "misfire", "confidence": 0.96},
        {"rul_estimate_timesteps": 45.71,  "rul_lower_bound_timesteps": 45.71,  "health_score": 96,  "predicted_fault": "misfire", "confidence": 1.0},
        {"rul_estimate_timesteps": 59.51,  "rul_lower_bound_timesteps": 59.51,  "health_score": 96,  "predicted_fault": "misfire", "confidence": 1.0},
    ]
    for i, reading in enumerate(fake_readings):
        smoothed = smoother.update(reading)
        print(f"step {i}: raw_rul_lb={reading['rul_lower_bound_timesteps']:<8} "
              f"smoothed_rul_lb={smoothed['rul_lower_bound_timesteps']} "
              f"raw_fault={reading['predicted_fault']:<8} "
              f"smoothed_fault={smoothed['predicted_fault']}")