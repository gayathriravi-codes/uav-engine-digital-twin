"""
augment_existing_flights.py -- adds battery_voltage and injection_timing_deg
(healthy baseline only) to every flight in data/raw/, so the existing 123
main-dataset flights have the same 9 sensor columns as the new fault-type
flights in data/raw_new_sensors/.

Does NOT touch the fault labels or values of any existing sensor -- these
two new columns get a healthy baseline only, since none of the 6 original
fault types affect the electrical/injection-timing subsystems (same
principle already used in fault_injectors.py: oil_issue doesn't touch EGT,
sensor_drift only touches its target sensor, etc.).

Writes augmented copies to data/raw_augmented/ -- originals in data/raw/
are never modified, so the existing leak-fixed classifier and its
xgboost_fault_classifier_fixed.json remain completely unaffected.

Run from repo root: python simulator/augment_existing_flights.py
"""
import os
import glob
import hashlib
import pandas as pd
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from battery_injection_sensors import attach_healthy_battery_injection_columns


def _seed_from_filename(filename):
    """Deterministic seed per file so reruns are reproducible."""
    h = hashlib.md5(filename.encode()).hexdigest()
    return int(h[:8], 16) % (2**31)


def augment_all(in_dir="data/raw", out_dir="data/raw_augmented"):
    os.makedirs(out_dir, exist_ok=True)
    csv_files = sorted(glob.glob(os.path.join(in_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {in_dir}")

    print(f"Found {len(csv_files)} flights in {in_dir}/")
    for i, filepath in enumerate(csv_files, start=1):
        filename = os.path.basename(filepath)
        df = pd.read_csv(filepath)
        seed = _seed_from_filename(filename)
        df = attach_healthy_battery_injection_columns(df, seed=seed)
        out_path = os.path.join(out_dir, filename)
        df.to_csv(out_path, index=False)
        if i % 20 == 0 or i == len(csv_files):
            print(f"[{i}/{len(csv_files)}] augmented -> {out_path}")

    print(f"\nDone. {len(csv_files)} augmented flights written to {out_dir}/")
    print("Originals in data/raw/ untouched -- existing model unaffected.")


if __name__ == "__main__":
    augment_all()
