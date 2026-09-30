"""
api/adapter.py -- stand-in adapter for apiserver.py.

apiserver.py imports:  FAULTS, HOOKS, iter_flight, new_state

IMPORTANT: this version is SELF-CONTAINED. It generates a synthetic flight
(sensor stream, fault, RUL, mission action) using only the Python standard
library. It does NOT call the trained XGBoost classifier, the RUL ensemble,
the sensor_trust checks or the real mission_reliability rules. Every tick
carries "source": "synthetic_adapter" so nobody mistakes it for real model
output. /health reports hooks_wired = false for every hook until you wire
the real modules in.

To wire a real module later, set an entry in HOOKS to a function:
    HOOKS["classifier"] = my_function
A hook is called as  my_function(tick, history)  and must return a dict of
fields that replace the same keys in the tick (for example
{"predicted_fault": "overheat", "confidence": 0.93}).
"""
import random

FAULTS = [
    "none", "misfire", "overheat", "cooling_degradation",
    "oil_issue", "sensor_drift", "vibration_fault",
]
SENSORS = ["rpm", "egt", "cht", "oil_pressure", "oil_temp", "fuel_flow", "vibration"]
ACTIONS = ["CONTINUE", "REDUCE_LOAD", "RTB", "ABORT"]

# Optional overrides. None = not wired (the synthetic value is used).
HOOKS = {
    "classifier": None,
    "rul": None,
    "mission": None,
    "sensor_trust": None,
    "shap": None,
}

N_TICKS = 300  # same length as the raw flight CSVs
RUL_CAP = 350

# Placeholder operating point and noise (NOT taken from real engine data).
BASE = {"rpm": 5200.0, "egt": 720.0, "cht": 150.0, "oil_pressure": 4.0,
        "oil_temp": 95.0, "fuel_flow": 18.0, "vibration": 0.15}
NOISE = {"rpm": 25.0, "egt": 4.0, "cht": 0.8, "oil_pressure": 0.04,
         "oil_temp": 0.5, "fuel_flow": 0.15, "vibration": 0.01}
# Size of a "large" deviation per sensor, used for drift and explanations.
SPREAD = {"rpm": 500.0, "egt": 90.0, "cht": 70.0, "oil_pressure": 1.8,
          "oil_temp": 25.0, "fuel_flow": 5.0, "vibration": 0.6}

# Detection delay (ticks after onset) at full severity, and smallest
# severity at which the stand-in classifier notices the fault at all.
DETECT_DELAY = {"vibration_fault": 49, "oil_issue": 52, "cooling_degradation": 54,
                "overheat": 59, "misfire": 124, "sensor_drift": 70}
MIN_DETECT_SEVERITY = {"vibration_fault": 0.35, "oil_issue": 0.35,
                       "cooling_degradation": 0.35, "overheat": 0.35,
                       "misfire": 0.75, "sensor_drift": 0.6}

FAULT_TEXT = {
    "misfire": "intermittent misfire pattern in RPM and EGT",
    "overheat": "cylinder head and exhaust temperatures rising",
    "cooling_degradation": "slow rise in cylinder head and oil temperature",
    "oil_issue": "oil pressure falling while oil temperature rises",
    "sensor_drift": "one sensor reading drifting or frozen",
    "vibration_fault": "vibration level rising",
}


def new_state():
    """Per-session state. apiserver.py stores operator confirmations here."""
    return {"confirmations": {}}


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def iter_flight(fault="none", severity=1.0, seed=None, state=None):
    """Yield one tick dict per timestep, N_TICKS in total."""
    if state is None:
        state = new_state()
    severity = _clamp(float(severity), 0.0, 1.0)
    rng = random.Random(seed if seed is not None else random.randrange(1_000_000))

    onset = 60 + rng.randint(0, 40)
    drift_sensor = rng.choice(SENSORS) if fault == "sensor_drift" else None
    flatline = fault == "sensor_drift" and rng.random() < 0.4
    frozen_value = None
    engine_fault = fault not in ("none", "sensor_drift")
    fail_tick = onset + int(90 + 200 * (1 - severity)) if engine_fault else None
    delay = 0
    if fault != "none":
        delay = int(DETECT_DELAY[fault] / max(severity, 0.35))

    history = []
    current_idx = 0     # index into ACTIONS after hysteresis
    down_streak = 0
    prev_action = "CONTINUE"

    for t in range(N_TICKS):
        ramp = 0.0
        if fault != "none" and t >= onset:
            ramp = min(1.0, (t - onset) / 60.0)
        r = ramp * severity

        s = {k: BASE[k] + rng.gauss(0, NOISE[k]) for k in SENSORS}

        # ---- fault effects on the true engine signals ----
        if fault == "overheat":
            s["cht"] += 70 * r
            s["egt"] += 90 * r
        elif fault == "cooling_degradation":
            s["cht"] += 40 * r
            s["oil_temp"] += 25 * r
        elif fault == "oil_issue":
            s["oil_pressure"] -= 1.8 * r
            s["oil_temp"] += 20 * r
        elif fault == "vibration_fault":
            s["vibration"] += 0.6 * r
        elif fault == "misfire":
            if rng.random() < 0.35 * ramp:
                s["rpm"] -= 500 * severity * rng.random()
                s["egt"] += 60 * severity
                s["vibration"] += 0.25 * severity
        elif fault == "sensor_drift" and t >= onset:
            if flatline:
                if frozen_value is None:
                    frozen_value = s[drift_sensor]
                s[drift_sensor] = frozen_value
            else:
                s[drift_sensor] += SPREAD[drift_sensor] * 0.6 * r

        # ---- sensor trust ----
        trust = {k: True for k in SENSORS}
        if fault == "sensor_drift" and t >= onset:
            if (flatline and t >= onset + 15) or ((not flatline) and r > 0.5):
                trust[drift_sensor] = False
        untrusted = [k for k in SENSORS if not trust[k]]

        # ---- health score (engine damage only; a bad sensor is not engine damage) ----
        damage = 0.0
        if engine_fault:
            damage = r * (0.8 if fault == "misfire" else 1.0)
        health = _clamp(100.0 - 85.0 * damage + rng.gauss(0, 1.0), 0.0, 100.0)
        if health >= 85:
            condition = "NORMAL"
        elif health >= 65:
            condition = "WATCH"
        elif health >= 40:
            condition = "DEGRADED"
        else:
            condition = "CRITICAL"

        # ---- fault classification (stand-in) ----
        detected = (fault != "none" and t >= onset + delay
                    and severity >= MIN_DETECT_SEVERITY[fault])
        if detected:
            predicted = fault
            confidence = _clamp(0.78 + 0.18 * min(1.0, (t - onset - delay) / 40.0)
                                + rng.gauss(0, 0.01), 0.0, 0.99)
        else:
            predicted = "none"
            confidence = _clamp(0.93 + rng.gauss(0, 0.02), 0.0, 0.99)
        rest = (1.0 - confidence) / (len(FAULTS) - 1)
        class_probs = {f: (confidence if f == predicted else rest) for f in FAULTS}

        # ---- RUL in timesteps (stand-in) ----
        if engine_fault and detected:
            rul_true = max(0, fail_tick - t)
        else:
            rul_true = RUL_CAP
        rul = _clamp(min(rul_true, RUL_CAP) * (1 + rng.gauss(0, 0.04)), 0, RUL_CAP)
        unc = 5.0 + 0.12 * rul
        rul_lower = max(0.0, rul - 1.3 * unc)

        # ---- mission action with simple hysteresis ----
        risk = _clamp((100.0 - health) / 85.0 * 0.7
                      + (1.0 - min(rul, RUL_CAP) / RUL_CAP) * 0.3, 0.0, 1.0)
        raw_idx = 0 if risk < 0.25 else 1 if risk < 0.5 else 2 if risk < 0.75 else 3
        if raw_idx > current_idx:
            current_idx = raw_idx
            down_streak = 0
        elif raw_idx < current_idx:
            down_streak += 1
            if down_streak >= 2:
                current_idx -= 1
                down_streak = 0
        else:
            down_streak = 0
        action = ACTIONS[current_idx]

        reasons = []
        if action == "CONTINUE":
            reasons.append("All monitored parameters are within normal bands")
        else:
            reasons.append("Health score %.0f (%s)" % (health, condition.lower()))
            if predicted != "none":
                reasons.append("Diagnosis: %s (%.0f%% confidence)"
                               % (predicted.replace("_", " "), confidence * 100))
            if rul < RUL_CAP:
                reasons.append("Remaining life about %.0f timesteps (lower bound %.0f)"
                               % (rul, rul_lower))
            if untrusted:
                reasons.append("Untrusted sensor: " + ", ".join(untrusted))

        requires_confirmation = action in ("RTB", "ABORT") and confidence < 0.85

        # ---- simple explanation (deviation of each sensor from its baseline) ----
        devs = {k: (s[k] - BASE[k]) / SPREAD[k] for k in SENSORS}
        top = sorted(devs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
        scale = max(1e-9, sum(abs(v) for _, v in top))
        explanation = [{"feature": k + "_deviation", "contribution": round(v / scale, 4)}
                       for k, v in top]

        tick = {
            "type": "tick",
            "source": "synthetic_adapter",
            "t": t,
            "tick": t,
            "mission_clock": "%02d:%02d" % (t // 60, t % 60),
            "sensors": {k: round(s[k], 4) for k in SENSORS},
            "trust": trust,
            "untrusted_sensors": untrusted,
            "health_score": round(health, 2),
            "condition_status": condition,
            "predicted_fault": predicted,
            "confidence": round(confidence, 4),
            "class_probs": {k: round(v, 4) for k, v in class_probs.items()},
            "severity": round(severity, 3),
            "rul_estimate_timesteps": round(rul, 1),
            "rul_lower_bound_timesteps": round(rul_lower, 1),
            "rul_uncertainty_timesteps": round(unc, 1),
            "recovery_rate": 0.0,
            "recovered": False,
            "action": action,
            "risk_score": round(risk, 3),
            "reasons": reasons,
            "action_changed": action != prev_action,
            "requires_confirmation": requires_confirmation,
            "confirmation": state["confirmations"].get(t),
            "explanation": explanation,
        }
        prev_action = action

        # ---- optional hooks: real modules can override fields ----
        for name, hook in HOOKS.items():
            if callable(hook):
                extra = hook(tick, history)
                if extra:
                    tick.update(extra)

        history.append(tick)
        yield tick
    # generator ends -> apiserver.py sends {"type": "end"}