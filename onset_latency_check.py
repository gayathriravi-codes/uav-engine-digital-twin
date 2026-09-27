"""
onset_latency_check.py
------------------------
Compares fault-onset detection latency between raw (unsmoothed) and
majority-vote-smoothed (MissionSmoother) fault_type predictions, across
a handful of flights per fault type.

For each flight: runs inference at increasing cutoffs (same START_ROW/STEP
as demo_playback.py), and records the first cutoff at which the predicted
fault_type matches the flight's true fault_type AND stays matched for the
rest of the flight (so a lone flickered true-positive tick doesn't count
as "detected" if it immediately reverts - only a real, sustained onset does).

Reports: true onset row, raw detection tick, smoothed detection tick,
and the gap in ticks each introduces relative to true onset.
"""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.classification.engine_inference import run_engine_inference
from mission_smoothing import MissionSmoother

START_ROW = 150
STEP = 5

# A handful of flights per validated fault type - edit this list to add more
FLIGHTS = [
    "data/raw/misfire_000.csv",
    "data/raw/misfire_001.csv",
    "data/raw/misfire_002.csv",
    "data/raw/oil_issue_000.csv",
    "data/raw/oil_issue_001.csv",
    "data/raw/oil_issue_002.csv",
    "data/raw/overheat_000.csv",
    "data/raw/overheat_001.csv",
    "data/raw/overheat_002.csv",
]


def first_sustained_match(ticks, predictions, true_fault):
    """First tick where predictions[i:] are ALL == true_fault (from i onward,
    ignoring any later flicker back to something else past that point isn't
    checked further here - this just finds first tick that starts an unbroken
    run to the end of the recorded predictions)."""
    for i in range(len(predictions)):
        if all(p == true_fault for p in predictions[i:]):
            return ticks[i]
    return None


def check_flight(flight_path):
    df = pd.read_csv(flight_path)
    fault_rows = df.index[df["fault_type"] != "none"]
    if len(fault_rows) == 0:
        # genuinely a healthy flight - skip onset comparison
        return {
            "flight": Path(flight_path).name,
            "true_fault": "none",
            "true_onset_idx": None,
            "raw_onset": None, "raw_gap": None,
            "smoothed_onset": None, "smoothed_gap": None,
            "extra_lag": None,
        }
    true_fault = df["fault_type"].iloc[fault_rows[0]]
    true_onset_idx = fault_rows[0]  # first row where the label actually reads as the fault
    smoother = MissionSmoother(window=3)
    ticks = []
    raw_preds = []
    smoothed_preds = []

    for cutoff in range(START_ROW, len(df) + 1, STEP):
        window_df = df.iloc[:cutoff].copy()
        engine_state = run_engine_inference(window_df)
        smoothed = smoother.update(engine_state)

        ticks.append(cutoff)
        raw_preds.append(engine_state["predicted_fault"])
        smoothed_preds.append(smoothed["predicted_fault"])

    raw_onset = first_sustained_match(ticks, raw_preds, true_fault)
    smoothed_onset = first_sustained_match(ticks, smoothed_preds, true_fault)

    raw_gap = (raw_onset - true_onset_idx) if raw_onset is not None else None
    smoothed_gap = (smoothed_onset - true_onset_idx) if smoothed_onset is not None else None

    return {
        "flight": Path(flight_path).name,
        "true_fault": true_fault,
        "true_onset_idx": true_onset_idx,
        "raw_onset": raw_onset,
        "raw_gap": raw_gap,
        "smoothed_onset": smoothed_onset,
        "smoothed_gap": smoothed_gap,
        "extra_lag": (smoothed_gap - raw_gap) if (raw_gap is not None and smoothed_gap is not None) else None,
    }


def main():
    print(f"{'flight':<20}{'true_fault':<12}{'true_onset':<12}{'raw_onset':<12}{'raw_gap':<10}{'smth_onset':<12}{'smth_gap':<10}{'extra_lag':<10}")
    print("-" * 98)
    results = []
    for flight_path in FLIGHTS:
        r = check_flight(flight_path)
        results.append(r)
        print(f"{r['flight']:<20}{r['true_fault']:<12}{r['true_onset_idx']:<12}"
              f"{str(r['raw_onset']):<12}{str(r['raw_gap']):<10}"
              f"{str(r['smoothed_onset']):<12}{str(r['smoothed_gap']):<10}"
              f"{str(r['extra_lag']):<10}")

    valid_extra_lags = [r["extra_lag"] for r in results if r["extra_lag"] is not None]
    if valid_extra_lags:
        avg_extra_lag = sum(valid_extra_lags) / len(valid_extra_lags)
        print("-" * 98)
        print(f"Average extra lag from smoothing: {avg_extra_lag:.1f} ticks "
              f"({avg_extra_lag * (STEP/5):.1f} timesteps at step={STEP})")


if __name__ == "__main__":
    main()