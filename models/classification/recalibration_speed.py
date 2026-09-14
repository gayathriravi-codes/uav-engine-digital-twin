"""
Engine health-score recalibration speed.

Measures how quickly the engine recovers after a meaningful
disturbance using fault-relevant sensor behaviour.

Output:
    {
        "recovery_rate": float,   # health points / timestep
        "recovered": bool
    }

The physical 0-100 health score is not modified.

Recovery detection is deliberately conservative:
    1. A meaningful disturbance must occur.
    2. Degradation must show a sustained rising trend.
    3. Candidate peaks are identified from trend reversals.
    4. The strongest meaningful candidate is evaluated first.
    5. A substantial fraction of the disturbance must recover.
    6. Recovery must remain stable for multiple windows.
    7. Intermittent faults require stronger confirmation.
"""

import numpy as np
import pandas as pd

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------

WINDOW_SIZE = 30
STRIDE = 5

# Number of consecutive windows required for confirmation.
MIN_RECOVERY_WINDOWS = 3

# Fraction of disturbance that must be recovered.
RECOVERY_FRACTION = 0.40

# Trend requirements around candidate peaks.
MIN_RISING_WINDOWS = 3
MIN_FALLING_WINDOWS = 2

# Ignore candidate peaks too close to the end of the flight.
END_MARGIN_WINDOWS = 3

# Minimum disturbance above baseline.
MIN_DISTURBANCE = 0.01

# Intermittent/mechanical faults are naturally more oscillatory.
# They therefore require stronger recovery confirmation.
STRICT_FAULTS = {
    "misfire",
    "vibration_fault",
    "sensor_drift",
}


# -------------------------------------------------------------
# Fault-specific sensor selection
# -------------------------------------------------------------

def _get_fault_type(df):
    """
    Determine the active fault type.

    Returns:
        str or None
    """

    if "fault_type" not in df.columns:
        return None

    fault_values = df["fault_type"].dropna()

    if len(fault_values) == 0:
        return None

    non_none = fault_values[
        fault_values != "none"
    ]

    if len(non_none) == 0:
        return None

    return str(non_none.iloc[0])


def _get_relevant_sensors(df):
    """
    Select sensors relevant to the detected fault.
    """

    fault_type = _get_fault_type(df)

    if fault_type is None:
        return []

    if fault_type == "overheat":
        return ["egt", "cht"]

    elif fault_type == "cooling_degradation":
        return ["cht"]

    elif fault_type == "oil_issue":
        return ["oil_pressure", "oil_temp"]

    elif fault_type == "vibration_fault":
        return ["vibration"]

    elif fault_type == "misfire":
        return ["rpm", "egt", "vibration"]

    elif fault_type == "sensor_drift":

        if (
            "drift_target_sensor" in df.columns
            and df["drift_target_sensor"].notna().any()
        ):
            target_sensor = (
                df["drift_target_sensor"]
                .dropna()
                .iloc[0]
            )

            if target_sensor in df.columns:
                return [target_sensor]

        return []

    return []


# -------------------------------------------------------------
# Build degradation signal
# -------------------------------------------------------------

def _build_degradation_signal(df):
    """
    Calculate relative deviation from the engine's own
    observed healthy baseline.

    Returns:
        numpy.ndarray
    """

    sensors = _get_relevant_sensors(df)

    if not sensors:
        return np.zeros(len(df))

    baseline_end = max(
        30,
        len(df) // 3
    )

    signals = []

    for sensor in sensors:

        if sensor not in df.columns:
            continue

        values = (
            df[sensor]
            .astype(float)
            .to_numpy()
        )

        baseline = float(
            np.mean(
                values[:baseline_end]
            )
        )

        if abs(baseline) < 1e-8:
            continue

        deviation = np.abs(
            (values - baseline)
            / abs(baseline)
        )

        signals.append(deviation)

    if not signals:
        return np.zeros(len(df))

    return np.mean(
        signals,
        axis=0
    )


# -------------------------------------------------------------
# Create rolling windows
# -------------------------------------------------------------

def _create_windows(df, signal):
    """
    Convert row-level degradation signal into
    30-row windows with stride 5.
    """

    window_values = []
    window_timestamps = []

    for start in range(
        0,
        len(df) - WINDOW_SIZE + 1,
        STRIDE
    ):

        end = start + WINDOW_SIZE

        window_values.append(
            float(
                np.mean(
                    signal[start:end]
                )
            )
        )

        window_timestamps.append(
            float(
                df["timestamp"]
                .iloc[end - 1]
            )
        )

    return (
        np.asarray(window_values),
        np.asarray(window_timestamps)
    )


# -------------------------------------------------------------
# Smooth temporal signal
# -------------------------------------------------------------

def _smooth_signal(values, window=3):
    """
    Light moving-average smoothing.

    This suppresses isolated fluctuations while preserving
    sustained degradation/recovery trends.
    """

    if len(values) < window:
        return values.copy()

    kernel = np.ones(window) / window

    return np.convolve(
        values,
        kernel,
        mode="valid"
    )


# -------------------------------------------------------------
# Find all meaningful recovery peaks
# -------------------------------------------------------------

def _find_recovery_candidates(
    window_degradation,
    baseline
):
    """
    Find all candidate disturbance peaks that satisfy
    sustained rising and falling trend requirements.

    Returns:
        List of tuples:
            (original_index, disturbance)
    """

    if len(window_degradation) < 10:
        return []

    smoothed = _smooth_signal(
        window_degradation,
        window=3
    )

    # Moving average shortens the signal by 2.
    offset = 1

    disturbance_threshold = max(
        baseline * 0.10,
        MIN_DISTURBANCE
    )

    max_candidate = (
        len(smoothed)
        - MIN_FALLING_WINDOWS
        - END_MARGIN_WINDOWS
    )

    candidates = []

    for i in range(
        MIN_RISING_WINDOWS,
        max_candidate
    ):

        original_idx = i + offset

        current = float(
            smoothed[i]
        )

        disturbance = (
            current - baseline
        )

        # -----------------------------------------------------
        # 1. Meaningful disturbance
        # -----------------------------------------------------

        if disturbance < disturbance_threshold:
            continue

        # -----------------------------------------------------
        # 2. Sustained rising trend
        # -----------------------------------------------------

        rising_segment = smoothed[
            i - MIN_RISING_WINDOWS:i + 1
        ]

        rising_diffs = np.diff(
            rising_segment
        )

        if not np.all(
            rising_diffs > 0
        ):
            continue

        # -----------------------------------------------------
        # 3. Sustained falling trend
        # -----------------------------------------------------

        falling_segment = smoothed[
            i:i + MIN_FALLING_WINDOWS + 1
        ]

        falling_diffs = np.diff(
            falling_segment
        )

        if not np.all(
            falling_diffs < 0
        ):
            continue

        candidates.append(
            (
                original_idx,
                disturbance
            )
        )

    return candidates


def _find_recovery_peak(
    window_degradation,
    baseline
):
    """
    Select the strongest meaningful recovery candidate.

    Returns:
        peak index in the ORIGINAL window_degradation array,
        or None.
    """

    candidates = _find_recovery_candidates(
        window_degradation,
        baseline
    )

    if not candidates:
        return None

    # Strongest disturbance first.
    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return candidates[0][0]


# -------------------------------------------------------------
# Main recalibration-speed calculation
# -------------------------------------------------------------

def compute_recalibration_speed(df):
    """
    Calculate engine recalibration speed.

    Returns:

        {
            "recovery_rate": float,
            "recovered": bool
        }

    recovery_rate is expressed as health points per timestep.
    """

    # ---------------------------------------------------------
    # Input validation
    # ---------------------------------------------------------

    if len(df) < WINDOW_SIZE:

        return {
            "recovery_rate": 0.0,
            "recovered": False
        }

    if "timestamp" not in df.columns:

        return {
            "recovery_rate": 0.0,
            "recovered": False
        }

    # Sort chronologically.
    data = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    fault_type = _get_fault_type(data)

    # ---------------------------------------------------------
    # Build temporal degradation signal
    # ---------------------------------------------------------

    degradation = _build_degradation_signal(
        data
    )

    if np.max(degradation) <= 1e-8:

        return {
            "recovery_rate": 0.0,
            "recovered": False
        }

    # ---------------------------------------------------------
    # Convert into rolling windows
    # ---------------------------------------------------------

    (
        window_degradation,
        window_timestamps
    ) = _create_windows(
        data,
        degradation
    )

    if len(window_degradation) < 10:

        return {
            "recovery_rate": 0.0,
            "recovered": False
        }

    # ---------------------------------------------------------
    # Healthy baseline
    # ---------------------------------------------------------

    baseline_count = max(
        3,
        len(window_degradation) // 3
    )

    baseline = float(
        np.mean(
            window_degradation[
                :baseline_count
            ]
        )
    )

    # ---------------------------------------------------------
    # Find strongest meaningful recovery peak
    # ---------------------------------------------------------

    peak_idx = _find_recovery_peak(
        window_degradation,
        baseline
    )

    if peak_idx is None:

        return {
            "recovery_rate": 0.0,
            "recovered": False
        }

    peak_degradation = float(
        window_degradation[
            peak_idx
        ]
    )

    peak_time = float(
        window_timestamps[
            peak_idx
        ]
    )

    # ---------------------------------------------------------
    # Calculate disturbance magnitude
    # ---------------------------------------------------------

    disturbance_amount = (
        peak_degradation
        - baseline
    )

    if disturbance_amount <= 0:

        return {
            "recovery_rate": 0.0,
            "recovered": False
        }

    # ---------------------------------------------------------
    # Define recovery target
    # ---------------------------------------------------------

    recovery_amount = (
        disturbance_amount
        * RECOVERY_FRACTION
    )

    recovery_target = (
        peak_degradation
        - recovery_amount
    )

    # ---------------------------------------------------------
    # Fault-specific confirmation
    # ---------------------------------------------------------

    if fault_type in STRICT_FAULTS:

        required_windows = (
            MIN_RECOVERY_WINDOWS + 1
        )

    else:

        required_windows = (
            MIN_RECOVERY_WINDOWS
        )

    # ---------------------------------------------------------
    # Find sustained recovery
    # ---------------------------------------------------------

    last_valid_start = (
        len(window_degradation)
        - required_windows
        - END_MARGIN_WINDOWS
    )

    for i in range(
        peak_idx + 1,
        max(
            peak_idx + 1,
            last_valid_start + 1
        )
    ):

        current_value = float(
            window_degradation[i]
        )

        if current_value > recovery_target:
            continue

        end_idx = (
            i
            + required_windows
        )

        if end_idx > len(window_degradation):
            break

        recovery_segment = (
            window_degradation[
                i:end_idx
            ]
        )

        # Recovery must persist.
        if not np.all(
            recovery_segment
            <= recovery_target
        ):
            continue

        # -----------------------------------------------------
        # Additional stability check for intermittent faults.
        # The first recovery point should not immediately
        # rebound above the target.
        # -----------------------------------------------------

        if fault_type in STRICT_FAULTS:

            stability_end = min(
                len(window_degradation),
                end_idx + 1
            )

            stability_segment = (
                window_degradation[
                    i:stability_end
                ]
            )

            if np.any(
                stability_segment
                > recovery_target
            ):
                continue

        # -----------------------------------------------------
        # Calculate recovery rate
        # -----------------------------------------------------

        recovered_idx = i

        recovered_degradation = float(
            window_degradation[
                recovered_idx
            ]
        )

        recovered_time = float(
            window_timestamps[
                recovered_idx
            ]
        )

        elapsed_timesteps = (
            recovered_time
            - peak_time
        )

        degradation_recovered = (
            peak_degradation
            - recovered_degradation
        )

        if (
            elapsed_timesteps <= 0
            or degradation_recovered <= 0
        ):

            continue

        recovery_rate = (
            degradation_recovered
            * 100.0
            / elapsed_timesteps
        )

        return {
            "recovery_rate": round(
                float(recovery_rate),
                4
            ),
            "recovered": True
        }

    # ---------------------------------------------------------
    # No recovery detected
    # ---------------------------------------------------------

    return {
        "recovery_rate": 0.0,
        "recovered": False
    }
