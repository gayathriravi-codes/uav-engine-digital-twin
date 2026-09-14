import os
import glob
import numpy as np
import pandas as pd

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from schema import SENSOR_FIELDS, WINDOW_SIZE, STRIDE, HEALTHY_RANGES

# ---------------------------------------------------------
# HEALTH SCORE
# ---------------------------------------------------------

SENSOR_WEIGHTS = {
    "rpm": 0.10,
    "egt": 0.20,
    "cht": 0.15,
    "oil_pressure": 0.15,
    "oil_temp": 0.10,
    "vibration": 0.15,
    "fuel_flow": 0.15,
}


def sensor_health(value, lower, upper):
    """
    Convert a sensor value into a 0-1 health score.

    1.0 = inside healthy operating range
    Lower values = degradation below range
    Higher values = degradation above range

    Overshoot is normalized by the healthy range's WIDTH
    (upper - lower), not by the raw threshold value. Dividing
    by the threshold itself under-penalizes sensors with large
    absolute healthy bounds (e.g. egt, cht) -- a fatal overshoot
    on those sensors was only producing a ~5-8% score drop.
    """

    range_width = upper - lower

    if lower <= value <= upper:
        return 1.0

    if value < lower:
        score = 1 - (lower - value) / range_width
    else:
        score = 1 - (value - upper) / range_width

    return np.clip(score, 0.0, 1.0)

def calculate_health_score(df):
    """
    Calculate interpretable engine health score for every row.

    Blends the weighted sensor average with the WORST single-sensor
    score, so one badly-degraded sensor can't be diluted away by six
    healthy ones (e.g. vibration_fault only affects vibration, but a
    pure average barely moves). Blend weights (0.6/0.4) are a first
    pass -- tune against real data if the split isn't right.

    Returns a pandas Series in the range 0-100.
    """

    scores = []

    for _, row in df.iterrows():

        weighted_score = 0.0
        worst_sensor_score = 1.0

        for sensor in SENSOR_FIELDS:
            lower, upper = HEALTHY_RANGES[sensor]

            h = sensor_health(
                row[sensor],
                lower,
                upper
            )

            weighted_score += SENSOR_WEIGHTS[sensor] * h
            worst_sensor_score = min(worst_sensor_score, h)

        blended = 0.6 * weighted_score + 0.4 * worst_sensor_score
        scores.append(100 * blended)

    return pd.Series(scores, index=df.index)

# ---------------------------------------------------------
# WINDOW FEATURES
# ---------------------------------------------------------

def calculate_slope(values):
    """
    Calculate linear trend/slope inside a window.
    """

    if len(values) < 2:
        return 0.0

    x = np.arange(len(values))

    return np.polyfit(x, values, 1)[0]


def extract_window_features(window):
    """
    Extract statistical and temporal features from one
    telemetry window.
    """

    features = {}

    # -----------------------------------------------------
    # Sensor-level statistical + temporal features
    # -----------------------------------------------------

    for sensor in SENSOR_FIELDS:

        values = window[sensor].astype(float).values

        features[f"{sensor}_mean"] = np.mean(values)
        features[f"{sensor}_std"] = np.std(values)
        features[f"{sensor}_min"] = np.min(values)
        features[f"{sensor}_max"] = np.max(values)
        features[f"{sensor}_range"] = np.max(values) - np.min(values)

        # Change from beginning to end of window
        features[f"{sensor}_delta"] = values[-1] - values[0]

        # Trend inside window
        features[f"{sensor}_slope"] = calculate_slope(values)

    # -----------------------------------------------------
    # Health-score features
    # -----------------------------------------------------

    health_scores = calculate_health_score(window)

    features["health_score_mean"] = health_scores.mean()
    features["health_score_std"] = health_scores.std()
    features["health_score_min"] = health_scores.min()
    features["health_score_delta"] = (
        health_scores.iloc[-1] - health_scores.iloc[0]
    )
    features["health_score_slope"] = calculate_slope(
        health_scores.values
    )

    return features


# ---------------------------------------------------------
# WINDOW LABEL
# ---------------------------------------------------------

def get_window_label(window):
    """
    Assign a fault label to a window.

    The majority label is used for normal fault types.

    Misfire is intermittent, so if a meaningful portion of
    the window contains misfire rows, classify the window
    as misfire.
    """

    counts = window["fault_type"].value_counts()

    # Special handling for intermittent misfire
    misfire_count = counts.get("misfire", 0)

    if misfire_count >= 5:
        return "misfire"

    # Otherwise use majority label
    return counts.idxmax()


# ---------------------------------------------------------
# PROCESS ONE FLIGHT
# ---------------------------------------------------------

def process_flight(filepath):

    df = pd.read_csv(filepath)

    # Sort by timestamp to guarantee temporal order
    df = df.sort_values("timestamp").reset_index(drop=True)

    rows = []

    for start in range(
        0,
        len(df) - WINDOW_SIZE + 1,
        STRIDE
    ):

        end = start + WINDOW_SIZE

        window = df.iloc[start:end]

        features = extract_window_features(window)

        features["label"] = get_window_label(window)

        features["flight_id"] = df["flight_id"].iloc[0]

        features["window_start"] = start
        features["window_end"] = end - 1

        rows.append(features)

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# PROCESS ENTIRE DATASET
# ---------------------------------------------------------

def build_feature_dataset(data_dir="data/raw"):

    csv_files = sorted(
        glob.glob(
            os.path.join(data_dir, "*.csv")
        )
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}"
        )

    all_features = []

    print(f"Found {len(csv_files)} flight files.")

    for i, filepath in enumerate(csv_files, start=1):

        print(
            f"[{i}/{len(csv_files)}] "
            f"Processing {os.path.basename(filepath)}"
        )

        flight_features = process_flight(filepath)

        all_features.append(flight_features)

    dataset = pd.concat(
        all_features,
        ignore_index=True
    )

    return dataset


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    dataset = build_feature_dataset()

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    output_path = (
        "data/processed/"
        "classification_features.csv"
    )

    dataset.to_csv(
        output_path,
        index=False
    )

    print("\nFeature extraction complete!")
    print(f"Saved to: {output_path}")

    print("\nDataset shape:")
    print(dataset.shape)

    print("\nLabel distribution:")
    print(dataset["label"].value_counts())

    print("\nFeature columns:")
    print(dataset.columns.tolist())