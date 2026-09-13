"""
preprocessing.py — slices each generated flight CSV into fixed-size,
overlapping windows for model input, and provides a leak-free
train/test split (split by flight, never by window).

Run directly: python simulator/preprocessing.py
"""
import os
import sys
import glob
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import SENSOR_FIELDS, WINDOW_SIZE, STRIDE


def create_windows(df, flight_id):
    """
    Slice one flight's dataframe into overlapping fixed-size windows.

    Each window becomes one training example:
      - features: (WINDOW_SIZE, n_sensor_fields) array
      - fault_type: label taken from the LAST row in the window
        (what the model would "see" up to that point)
      - true_rul: ground-truth remaining-useful-life at the last row
        in the window (-1 sentinel for healthy-only flights)
      - flight_id: which flight this window came from (needed for the
        leak-free split -- windows from the same flight must never be
        split across train/test)
    """
    windows = []
    n_rows = len(df)
    if n_rows < WINDOW_SIZE:
        return windows

    sensor_data = df[SENSOR_FIELDS].to_numpy(dtype=np.float32)
    fault_types = df["fault_type"].to_numpy()
    true_ruls = df["true_rul_timesteps"].to_numpy()

    for start in range(0, n_rows - WINDOW_SIZE + 1, STRIDE):
        end = start + WINDOW_SIZE
        windows.append({
            "features": sensor_data[start:end],
            "fault_type": fault_types[end - 1],
            "true_rul": float(true_ruls[end - 1]),
            "flight_id": flight_id,
        })
    return windows


def build_dataset(data_dir):
    """
    Load every flight CSV in data_dir, window each one, and return:
      - all_windows: list of window dicts (see create_windows)
      - flight_ids: parallel list, one flight_id per window
    """
    csv_paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_paths:
        print(f"No CSV files found in {data_dir} -- "
              f"run fault_injectors.py first to generate flights.")

    all_windows = []
    flight_ids = []

    for path in csv_paths:
        df = pd.read_csv(path).sort_values("timestamp").reset_index(drop=True)
        flight_id = df["flight_id"].iloc[0]
        windows = create_windows(df, flight_id)
        all_windows.extend(windows)
        flight_ids.extend([flight_id] * len(windows))

    return all_windows, flight_ids


def train_test_split_by_flight(all_windows, flight_ids, test_size=0.2, seed=42):
    """
    Split by FLIGHT, not by window -- otherwise windows from the same
    flight (which overlap and share most of their timesteps) end up on
    both sides of the split, and the model gets an artificially easy
    test set. Every window from a given flight goes entirely to train
    or entirely to test.
    """
    unique_flights = sorted(set(flight_ids))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_flights)

    n_test = max(1, int(len(unique_flights) * test_size))
    test_flights = set(unique_flights[:n_test])
    train_flights = set(unique_flights[n_test:])

    train_windows = [w for w, fid in zip(all_windows, flight_ids) if fid in train_flights]
    test_windows = [w for w, fid in zip(all_windows, flight_ids) if fid in test_flights]

    overlap = train_flights & test_flights
    assert not overlap, f"Leak-free split violated -- flights in both sets: {overlap}"

    return train_windows, test_windows


if __name__ == "__main__":
    print("Building windowed dataset from data/raw ...")
    all_windows, flight_ids = build_dataset("data/raw")
    print(f"Total windows: {len(all_windows)}  |  Total flights: {len(set(flight_ids))}")

    counts = Counter(w["fault_type"] for w in all_windows)
    print("\nWindow counts by fault type:")
    for fault_type, count in counts.most_common():
        print(f"  {fault_type:<20} {count}")

    train_w, test_w = train_test_split_by_flight(all_windows, flight_ids)
    train_flights = set(w["flight_id"] for w in train_w)
    test_flights = set(w["flight_id"] for w in test_w)

    print(f"\nTrain windows: {len(train_w)}  ({len(train_flights)} flights)")
    print(f"Test windows:  {len(test_w)}  ({len(test_flights)} flights)")
    print(f"Zero-overlap confirmed: {len(train_flights & test_flights) == 0}")