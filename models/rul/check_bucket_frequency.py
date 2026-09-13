"""
check_bucket_frequency.py -- how often do REAL windows land in each of
Ashmitha's calibration buckets, especially (250,350) (the one with zero
validation windows, where calibration is a documented no-op)?

Two populations checked:
  1. Raw dataset windows -- every window from every flight in data/raw,
     using the CALIBRATED point estimate (what a user would actually see).
  2. What-if scenario windows -- the counterfactual continuations the
     what-if engine generates (severity 0.35/0.45/0.5/1.0), since those are
     regenerated data and could land in different ranges than the raw
     dataset -- this is the population that actually matters most for
     deciding whether the (250,350) gap is worth fixing.

Run: python models/rul/check_bucket_frequency.py   (from project root)
"""
import os
import sys
import glob

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from schema import SENSOR_FIELDS, WINDOW_SIZE

from train_rul_ensemble import build_inference_window, predict_rul_ensemble, load_ensemble

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "simulator"))
from simulate_healthy import simulate_healthy_flight  # noqa: E402
from fault_injectors import INJECTORS  # noqa: E402

BUCKETS = [(0, 50), (50, 100), (100, 200), (200, 250), (250, 350)]


def bucket_of(value):
    for lo, hi in BUCKETS:
        if lo <= value < hi:
            return f"({lo},{hi})"
    return "out_of_range" if value < BUCKETS[0][0] else f">={BUCKETS[-1][1]}"


def check_raw_dataset_windows(models, scaler, dropped_idx_list, data_dir="data/raw", stride_sample=1):
    """Buckets the CALIBRATED point estimate for every window in every flight."""
    csv_paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    counts = {f"({lo},{hi})": 0 for lo, hi in BUCKETS}
    total = 0

    for path in csv_paths:
        df = pd.read_csv(path)
        if df["true_rul_timesteps"].iloc[0] == -1.0 and df["fault_type"].iloc[0] == "none":
            continue  # skip healthy-only flights, same exclusion as training
        sensor_data = df[SENSOR_FIELDS].values
        for start in range(0, len(df) - WINDOW_SIZE + 1, stride_sample):
            raw_window = sensor_data[start:start + WINDOW_SIZE]
            featurized = build_inference_window(raw_window)
            result = predict_rul_ensemble(featurized, models, scaler, dropped_idx_list, calibrated=True)
            b = bucket_of(result["point_estimate_minutes"])
            if b in counts:
                counts[b] += 1
            total += 1

    return counts, total


def check_whatif_scenario_windows(models, scaler, dropped_idx_list, data_dir="data/raw"):
    """
    Buckets the CALIBRATED point estimate across all 4 what-if severity
    scenarios, run at several onset-relative positions per fault type --
    this is the population the what-if engine actually generates live.
    """
    csv_paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    sample_flights = {}
    for path in csv_paths:
        df = pd.read_csv(path)
        ft = df["fault_type"].iloc[-1]
        if ft != "none" and ft not in sample_flights:
            sample_flights[ft] = (path, df)

    severities = [1.0, 0.35, 0.45, 0.5]  # continue_current + 3 intervention scenarios
    position_fracs = [0.1, 0.3, 0.5, 0.7, 0.9]  # spread across the fault's progression

    counts = {f"({lo},{hi})": 0 for lo, hi in BUCKETS}
    total = 0

    for fault_type, (path, df) in sample_flights.items():
        onset_idx = int(df["fault_onset_idx"].iloc[0])
        duration = len(df)
        for frac in position_fracs:
            window_start = onset_idx + int(frac * (duration - onset_idx))
            window_start = min(window_start, duration - WINDOW_SIZE)
            if window_start < 0:
                continue
            for severity in severities:
                healthy = simulate_healthy_flight(duration_timesteps=duration, seed=42)
                onset_frac = onset_idx / max(1, duration - 1)
                counterfactual = INJECTORS[fault_type](healthy, onset_frac=onset_frac, severity=severity, seed=42)
                raw_window = counterfactual[SENSOR_FIELDS].to_numpy(dtype=np.float32)[
                    window_start:window_start + WINDOW_SIZE
                ]
                featurized = build_inference_window(raw_window)
                result = predict_rul_ensemble(featurized, models, scaler, dropped_idx_list, calibrated=True)
                b = bucket_of(result["point_estimate_minutes"])
                if b in counts:
                    counts[b] += 1
                total += 1

    return counts, total


if __name__ == "__main__":
    print("Loading trained ensemble ...")
    models, scaler, dropped_idx_list = load_ensemble()

    print("\n=== Population 1: every window in data/raw (fault flights only) ===")
    counts1, total1 = check_raw_dataset_windows(models, scaler, dropped_idx_list)
    for b, c in counts1.items():
        pct = 100 * c / total1 if total1 else 0
        print(f"  {b:<12} {c:5d} windows  ({pct:5.1f}%)")
    print(f"  TOTAL: {total1} windows")

    print("\n=== Population 2: what-if scenario windows (all 6 fault types x 5 positions x 4 severities) ===")
    counts2, total2 = check_whatif_scenario_windows(models, scaler, dropped_idx_list)
    for b, c in counts2.items():
        pct = 100 * c / total2 if total2 else 0
        print(f"  {b:<12} {c:5d} windows  ({pct:5.1f}%)")
    print(f"  TOTAL: {total2} windows")

    print("\n=== Verdict ===")
    p1_share = 100 * counts1["(250,350)"] / total1 if total1 else 0
    p2_share = 100 * counts2["(250,350)"] / total2 if total2 else 0
    print(f"(250,350) bucket share: {p1_share:.1f}% of raw dataset windows, "
          f"{p2_share:.1f}% of what-if scenario windows.")
    if p1_share < 1 and p2_share < 1:
        print("VERY RARE in practice -- documenting as a known limitation is reasonable, "
              "not worth spending more time fixing before the hackathon.")
    else:
        print("NOT rare enough to ignore -- worth a real fix (e.g. merge (250,350) into "
              "(200,250) the same way (100,150)/(150,200) were merged in v5) before relying "
              "on this range in a live demo.")