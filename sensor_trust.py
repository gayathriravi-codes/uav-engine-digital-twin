"""
sensor_trust.py
-----------------
Sensor-trust flag: flags individual sensors within a telemetry window
as untrustworthy (impossible value, flatlined/stuck, an impossible
rate-of-change jump, or a sustained mean-shift) rather than treating
every out-of-range reading as a genuine engine fault.

Why: directly extends the sensor_drift diagnosis. If a sensor is lying,
no amount of downstream modeling on its readings is meaningful - the
right response is to flag the sensor itself as untrustworthy, rather
than let a broken sensor silently masquerade as a real fault or as
healthy operation.

Four independent checks, each catching a different failure mode:
  - impossible_value: reading outside physically possible bounds
  - flatlined: suspiciously constant (stuck sensor)
  - impossible_rate_of_change: an implausible single-tick jump
  - sustained_mean_shift: a persistent bias/step-change that's too
    small to catch via single-tick rate-of-change (can be smaller than
    normal tick-to-tick noise, but still detectable once averaged over
    enough ticks) - added after finding a real bias-type sensor_drift
    fault on vibration where the injected jump (0.287) was BELOW the
    sensor's own normal max tick-to-tick delta (0.337), meaning
    single-tick detection alone cannot reliably catch it.

This module is intentionally independent of the ML fault classifier -
it runs simple, fully-inspectable rule checks against physical bounds
in schema.py, so it's easy to explain and defend in Q&A.
"""

import numpy as np

from schema import (
    SENSOR_FIELDS,
    PHYSICAL_LIMITS,
    MAX_RATE_OF_CHANGE,
    FLATLINE_STD_THRESHOLD,
)

# Sustained mean-shift detection config: compares the mean of the most
# recent `recent_window` readings against the mean of the
# `baseline_window` readings before that, flagging a shift larger than
# `threshold_std_multiples` times the baseline window's own std.
MEAN_SHIFT_CONFIG = {
    "rpm": {"recent_window": 10, "baseline_window": 20, "threshold_std_multiples": 3.0},
    "egt": {"recent_window": 10, "baseline_window": 20, "threshold_std_multiples": 3.0},
    "cht": {"recent_window": 10, "baseline_window": 20, "threshold_std_multiples": 3.0},
    "oil_pressure": {"recent_window": 10, "baseline_window": 20, "threshold_std_multiples": 3.0},
    "oil_temp": {"recent_window": 10, "baseline_window": 20, "threshold_std_multiples": 3.0},
    "vibration": {"recent_window": 10, "baseline_window": 20, "threshold_std_multiples": 3.0},
    "fuel_flow": {"recent_window": 10, "baseline_window": 20, "threshold_std_multiples": 3.0},
}


def check_impossible_value(values, sensor):
    """Any reading outside PHYSICAL_LIMITS - the sensor itself is lying,
    not just reporting a real degraded engine state."""
    lower, upper = PHYSICAL_LIMITS[sensor]
    return bool(np.any((values < lower) | (values > upper)))


def check_flatline(values, sensor):
    """Suspiciously constant readings - a stuck sensor, not genuinely
    stable engine behavior."""
    return bool(np.std(values) < FLATLINE_STD_THRESHOLD[sensor])


def check_rate_of_change(values, sensor):
    """A jump between consecutive readings larger than physically
    plausible - a glitch/dropout, not real engine dynamics."""
    if len(values) < 2:
        return False
    diffs = np.abs(np.diff(values))
    return bool(np.any(diffs > MAX_RATE_OF_CHANGE[sensor]))


def check_sustained_mean_shift(values, sensor):
    """
    Compares the mean of the most recent `recent_window` readings against
    the mean of the `baseline_window` readings immediately before that,
    flagging a shift larger than `threshold_std_multiples` times the
    baseline window's own std. Catches a persistent bias/step-change even
    when it's individually smaller than normal tick-to-tick noise - a
    single noisy tick averages out, but a real sustained shift does not.
    """
    cfg = MEAN_SHIFT_CONFIG[sensor]
    recent_n = cfg["recent_window"]
    baseline_n = cfg["baseline_window"]
    total_needed = recent_n + baseline_n

    if len(values) < total_needed:
        return False  # not enough history yet to compare windows

    baseline = values[-total_needed:-recent_n]
    recent = values[-recent_n:]

    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline)
    recent_mean = np.mean(recent)

    if baseline_std < 1e-6:
        return bool(abs(recent_mean - baseline_mean) > 1e-6)

    shift_in_stds = abs(recent_mean - baseline_mean) / baseline_std
    return bool(shift_in_stds > cfg["threshold_std_multiples"])


def evaluate_sensor_trust(window):
    """
    Evaluate sensor trustworthiness for one telemetry window.

    Parameters
    ----------
    window : pandas.DataFrame
        A telemetry window containing sensor columns from SENSOR_FIELDS.
        Should contain enough rows to cover the mean-shift check's
        recent_window + baseline_window (30 rows by default config) -
        rows short of that just skip the mean-shift check, not error.

    Returns
    -------
    dict
        {
          "trusted_sensors": [...],
          "untrusted_sensors": [...],
          "flags": {sensor: [list of failed checks]},
          "any_untrusted": bool,
        }
    """
    flags = {}
    untrusted_sensors = []
    trusted_sensors = []

    for sensor in SENSOR_FIELDS:
        values = window[sensor].astype(float).values
        sensor_flags = []

        if check_impossible_value(values, sensor):
            sensor_flags.append("impossible_value")

        if check_flatline(values, sensor):
            sensor_flags.append("flatlined")

        if check_rate_of_change(values, sensor):
            sensor_flags.append("impossible_rate_of_change")

        if check_sustained_mean_shift(values, sensor):
            sensor_flags.append("sustained_mean_shift")

        if sensor_flags:
            flags[sensor] = sensor_flags
            untrusted_sensors.append(sensor)
        else:
            trusted_sensors.append(sensor)

    return {
        "trusted_sensors": trusted_sensors,
        "untrusted_sensors": untrusted_sensors,
        "flags": flags,
        "any_untrusted": len(untrusted_sensors) > 0,
    }


if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path

    csv_path = Path("data/raw/sensor_drift_000.csv")
    df = pd.read_csv(csv_path)
    df = df.sort_values("timestamp").reset_index(drop=True)

    onset = df.index[df["fault_type"] != "none"][0]
    window = df.iloc[max(0, onset - 30):onset + 15].copy()
    result = evaluate_sensor_trust(window)

    print("=" * 60)
    print(f"SENSOR TRUST CHECK - {csv_path.name} (window straddling onset at row {onset})")
    print("=" * 60)
    print(f"Trusted sensors   : {result['trusted_sensors']}")
    print(f"Untrusted sensors : {result['untrusted_sensors']}")
    print(f"Any untrusted     : {result['any_untrusted']}")
    if result["flags"]:
        print("\nFlag details:")
        for sensor, sensor_flags in result["flags"].items():
            print(f"  {sensor}: {sensor_flags}")