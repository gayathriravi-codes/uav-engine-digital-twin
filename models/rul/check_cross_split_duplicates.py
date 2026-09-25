"""
check_cross_split_duplicates.py

Does the (0009/0012/0016/etc-style) trajectory duplication found within val
ALSO cross the train/val/test boundary? Uses the same window-0 fingerprint
technique, but computed across ALL flights (train+val+test) before any split,
then checks which split each duplicate group's members landed in.

Read-only. Does not touch test predictions/labels, only flight identity.
Run: python check_cross_split_duplicates.py
"""
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rul import load_all_flights
from train_rul_ensemble import build_windowed_dataset_v5


def main():
    print("Loading flights and building windowed dataset ...")
    flights = load_all_flights()
    X, y, flight_ids = build_windowed_dataset_v5(flights)

    unique_flights = np.unique(flight_ids)
    train_flights, temp_flights = train_test_split(unique_flights, test_size=0.3, random_state=42)
    val_flights, test_flights = train_test_split(temp_flights, test_size=0.5, random_state=42)

    split_of = {}
    for f in train_flights:
        split_of[f] = "train"
    for f in val_flights:
        split_of[f] = "val"
    for f in test_flights:
        split_of[f] = "test"

    print(f"Total unique flights: {len(unique_flights)} "
          f"(train={len(train_flights)}, val={len(val_flights)}, test={len(test_flights)})")

    print("\nFingerprinting ALL flights (window-0 per-channel means, full precision) ...")
    fingerprints = {}
    for fid in unique_flights:
        first_window = X[flight_ids == fid][0]
        fp = tuple(np.round(first_window.mean(axis=0), 3))
        fingerprints.setdefault(fp, []).append(fid)

    print("\n=== Duplicate groups and which split(s) they span ===")
    cross_split_groups = 0
    total_dup_flights = 0
    for fp, fids in fingerprints.items():
        if len(fids) <= 1:
            continue
        total_dup_flights += len(fids)
        splits_hit = sorted(set(split_of[f] for f in fids))
        flag = "  <-- CROSSES SPLITS" if len(splits_hit) > 1 else ""
        if len(splits_hit) > 1:
            cross_split_groups += 1
        print(f"  {fids}  splits={splits_hit}{flag}")

    print(f"\nTotal flights involved in any duplicate group: {total_dup_flights} / {len(unique_flights)}")
    print(f"Duplicate groups that CROSS train/val/test: {cross_split_groups}")
    print("\nIf cross_split_groups > 0: train/val/test are not independent for those")
    print("flights -- every metric computed this session (OOD gate, reweighting,")
    print("classifier) needs to be treated as unreliable until re-evaluated on a")
    print("split that groups duplicate-trajectory flights together on one side.")
    print("If cross_split_groups == 0: the duplication exists but happens to be")
    print("split-safe by luck of this particular random_state=42 partition --")
    print("still worth fixing properly (group by base trajectory, not flight_id)")
    print("before it silently breaks a future reshuffle.")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()