"""
feature_extraction_v2.py -- extends models/classification/feature_extraction.py
with:

  1. Window stats (mean/std/min/max/range/delta/slope) for the two new
     sensors (battery_voltage, injection_timing_deg), same as the existing
     7 core sensors.

  2. A cross-sensor CORRELATION-COLLAPSE feature: in a healthy engine,
     rpm<->fuel_flow and egt<->cht move together predictably. This computes
     each pair's correlation WITHIN each window, and the absolute deviation
     from a healthy reference baseline correlation (computed once from a
     simulated healthy flight). A fault that breaks the engine's normal
     physical coupling (e.g. injector_abnormality: fuel_flow drops but egt
     doesn't follow proportionally) shows up as a correlation collapse --
     a signature no single-sensor mean/std/slope feature can see.

Does NOT modify feature_extraction.py or its output file -- writes to a
separate classification_features_v2.csv so the original leak-fixed
91.1% result and its saved model stay completely intact as a fallback.

Run from repo root: python models/classification/feature_extraction_v2.py
(after running simulator/augment_existing_flights.py first)
"""
import os
import glob
import numpy as np
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "simulator"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `from feature_extraction import ...`

from schema import SENSOR_FIELDS, WINDOW_SIZE, STRIDE, HEALTHY_RANGES
from feature_extraction import (
    SENSOR_WEIGHTS,
    sensor_health,
    calculate_health_score,
    calculate_slope,
    get_window_label,
)
from simulate_healthy import simulate_healthy_flight
from battery_injection_sensors import attach_healthy_battery_injection_columns

# ---------------------------------------------------------
# NEW SENSORS
# ---------------------------------------------------------
NEW_SENSOR_FIELDS = ["battery_voltage", "injection_timing_deg"]
ALL_SENSOR_FIELDS = SENSOR_FIELDS + NEW_SENSOR_FIELDS

# ---------------------------------------------------------
# CROSS-SENSOR CORRELATION BASELINE
# ---------------------------------------------------------
# Pairs chosen because they're physically coupled in a healthy engine:
#   rpm <-> fuel_flow: more RPM needs more fuel, roughly proportional
#   egt <-> cht: both track combustion heat, normally move together
CORRELATION_PAIRS = [
    ("rpm", "fuel_flow"),
    ("egt", "cht"),
    # Added specifically to target combustion_instability: this fault injects
    # a SHARED jitter signal onto both rpm and egt (see
    # combustion_instability_injector.py's inject_combustion_instability),
    # so a rpm<->egt correlation feature is the one most directly aimed at
    # its actual design, distinct from the rpm<->fuel_flow /egt<->cht pairs
    # already tested (and which the ablation study found did not help).
    ("rpm", "egt"),
]


def _compute_baseline_correlations(n_reference_flights=5, duration=300):
    """
    Computes each pair's correlation on WINDOW_SIZE-length windows from
    several simulated healthy flights, averaged, so the baseline isn't
    an artifact of one specific random seed's noise.
    """
    baselines = {pair: [] for pair in CORRELATION_PAIRS}
    for seed in range(n_reference_flights):
        healthy = simulate_healthy_flight(duration_timesteps=duration, seed=90000 + seed)
        for start in range(0, len(healthy) - WINDOW_SIZE + 1, STRIDE):
            window = healthy.iloc[start:start + WINDOW_SIZE]
            for pair in CORRELATION_PAIRS:
                a, b = pair
                va, vb = window[a].astype(float).values, window[b].astype(float).values
                if np.std(va) < 1e-8 or np.std(vb) < 1e-8:
                    continue
                baselines[pair].append(np.corrcoef(va, vb)[0, 1])
    return {pair: float(np.mean(vals)) for pair, vals in baselines.items()}


print("Computing healthy baseline correlations from reference flights...")
BASELINE_CORRELATIONS = _compute_baseline_correlations()
for pair, val in BASELINE_CORRELATIONS.items():
    print(f"  baseline corr{pair}: {val:.3f}")


def _window_correlation(window, sensor_a, sensor_b):
    va = window[sensor_a].astype(float).values
    vb = window[sensor_b].astype(float).values
    if np.std(va) < 1e-8 or np.std(vb) < 1e-8:
        # A flatlined sensor has no correlation to measure -- treat as
        # maximal collapse (0.0) rather than NaN, since a flatline itself
        # is already a strong abnormality signal.
        return 0.0
    corr = np.corrcoef(va, vb)[0, 1]
    return 0.0 if np.isnan(corr) else corr


# ---------------------------------------------------------
# EXTENDED WINDOW FEATURES
# ---------------------------------------------------------

def extract_window_features_v2(window):
    features = {}

    for sensor in ALL_SENSOR_FIELDS:
        if sensor not in window.columns:
            continue
        values = window[sensor].astype(float).values
        features[f"{sensor}_mean"] = np.mean(values)
        features[f"{sensor}_std"] = np.std(values)
        features[f"{sensor}_min"] = np.min(values)
        features[f"{sensor}_max"] = np.max(values)
        features[f"{sensor}_range"] = np.max(values) - np.min(values)
        features[f"{sensor}_delta"] = values[-1] - values[0]
        features[f"{sensor}_slope"] = calculate_slope(values)

    # Health-score features unchanged -- still computed only from the 7
    # core sensors, since SENSOR_WEIGHTS/HEALTHY_RANGES are only defined
    # for those (the two new sensors don't have a defined "health"
    # semantics the way the core 7 do).
    health_scores = calculate_health_score(window)
    features["health_score_mean"] = health_scores.mean()
    features["health_score_std"] = health_scores.std()
    features["health_score_min"] = health_scores.min()
    features["health_score_delta"] = health_scores.iloc[-1] - health_scores.iloc[0]
    features["health_score_slope"] = calculate_slope(health_scores.values)

    # Cross-sensor correlation-collapse features
    for pair in CORRELATION_PAIRS:
        a, b = pair
        window_corr = _window_correlation(window, a, b)
        baseline = BASELINE_CORRELATIONS[pair]
        features[f"corr_{a}_{b}"] = window_corr
        features[f"corr_{a}_{b}_collapse"] = abs(baseline - window_corr)

    return features


# ---------------------------------------------------------
# PROCESS ONE FLIGHT
# ---------------------------------------------------------

def process_flight_v2(filepath):
    df = pd.read_csv(filepath)
    df = df.sort_values("timestamp").reset_index(drop=True)

    rows = []
    for start in range(0, len(df) - WINDOW_SIZE + 1, STRIDE):
        end = start + WINDOW_SIZE
        window = df.iloc[start:end]
        features = extract_window_features_v2(window)
        features["label"] = get_window_label(window)
        features["flight_id"] = df["flight_id"].iloc[0]
        features["window_start"] = start
        features["window_end"] = end - 1
        rows.append(features)

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# PROCESS MULTIPLE DATA DIRECTORIES
# ---------------------------------------------------------

def build_feature_dataset_v2(data_dirs=("data/raw_augmented", "data/raw_new_sensors")):
    all_features = []
    total_files = 0

    for data_dir in data_dirs:
        csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        if not csv_files:
            print(f"WARNING: no CSV files found in {data_dir} -- skipping")
            continue
        print(f"Found {len(csv_files)} flight files in {data_dir}/")
        for i, filepath in enumerate(csv_files, start=1):
            total_files += 1
            if i % 20 == 0 or i == len(csv_files):
                print(f"  [{i}/{len(csv_files)}] processing {os.path.basename(filepath)}")
            all_features.append(process_flight_v2(filepath))

    if not all_features:
        raise FileNotFoundError(f"No CSV files found in any of {data_dirs}")

    dataset = pd.concat(all_features, ignore_index=True)
    print(f"\nTotal flights processed: {total_files}")
    return dataset


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    dataset = build_feature_dataset_v2()

    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/classification_features_v2.csv"
    dataset.to_csv(output_path, index=False)

    print("\nFeature extraction (v2) complete!")
    print(f"Saved to: {output_path}")
    print("\nDataset shape:", dataset.shape)
    print("\nLabel distribution:")
    print(dataset["label"].value_counts())