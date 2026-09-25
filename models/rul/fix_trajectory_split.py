"""
fix_trajectory_split.py

Fixes the train/val/test split leakage discovered this session: every
flight_id is one of 18 base trajectories (identified by trailing _NNN),
relabeled under 6 fault-type names each (cooling_degradation_NNN,
misfire_NNN, oil_issue_NNN, overheat_NNN, sensor_drift_NNN,
vibration_fault_NNN -- all byte-identical for the first ~15-30 windows,
diverging only as a fault effect ramps up). The OLD split called
train_test_split() on the 108 fault-labeled names directly, so copies of
the same base trajectory routinely landed in train AND val AND/OR test
(15 / 18 base trajectories crossed splits; 108/108 flights were part of
a duplicate group).

This script:
  1. Groups flight_ids by base trajectory ID (everything after the last
     underscore), NOT by the full fault-labeled name.
  2. Splits on the 18 base trajectory IDs instead of the 108 flight_ids,
     so every fault-labeled copy of a given base trajectory lands in the
     SAME split -- no more leakage.
  3. Re-runs check_per_bucket_coverage() on the corrected val split (the
     val-side analogue of run_coverage_check.py) so we get an honest
     number before ever touching test again.

This intentionally does NOT touch run_coverage_check.py / test yet --
see printed guidance at the end for that step.

Run: python fix_trajectory_split.py
"""
import os
import re
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rul import load_all_flights
from train_rul_ensemble import build_windowed_dataset_v5, load_ensemble
from calibrate_rul import load_calibration_params, check_per_bucket_coverage


def base_trajectory_id(flight_id):
    """'sensor_drift_009' -> '009'. Adjust the regex if the naming
    convention turns out to have more than one trailing numeric group
    or non-numeric suffixes -- inspect a few flight_ids first if this
    assertion fails."""
    m = re.search(r'_(\d+)$', flight_id)
    if not m:
        raise ValueError(
            f"flight_id {flight_id!r} doesn't match the expected "
            f"'<fault_type>_<NNN>' pattern -- inspect naming before trusting this split."
        )
    return m.group(1)


def main():
    print("Loading flights and building windowed dataset ...")
    flights = load_all_flights()
    X, y, flight_ids = build_windowed_dataset_v5(flights)

    unique_flights = np.unique(flight_ids)
    base_ids = np.array([base_trajectory_id(f) for f in unique_flights])
    unique_bases = np.unique(base_ids)
    print(f"Unique flight_ids: {len(unique_flights)}  ->  unique BASE trajectories: {len(unique_bases)}")

    # --- OLD (leaky) split, for side-by-side comparison ---
    old_train_flights, old_temp_flights = train_test_split(unique_flights, test_size=0.3, random_state=42)
    old_val_flights, old_test_flights = train_test_split(old_temp_flights, test_size=0.5, random_state=42)
    old_split_of = {f: "train" for f in old_train_flights}
    old_split_of.update({f: "val" for f in old_val_flights})
    old_split_of.update({f: "test" for f in old_test_flights})

    # --- NEW split: split on BASE trajectory IDs, then propagate to every
    #     fault-labeled flight_id sharing that base ---
    train_bases, temp_bases = train_test_split(unique_bases, test_size=0.3, random_state=42)
    val_bases, test_bases = train_test_split(temp_bases, test_size=0.5, random_state=42)

    train_bases_set, val_bases_set, test_bases_set = set(train_bases), set(val_bases), set(test_bases)

    def base_of(fid):
        return base_trajectory_id(fid)

    train_flights = np.array([f for f in unique_flights if base_of(f) in train_bases_set])
    val_flights = np.array([f for f in unique_flights if base_of(f) in val_bases_set])
    test_flights = np.array([f for f in unique_flights if base_of(f) in test_bases_set])

    print(f"\nNEW split (grouped by base trajectory): "
          f"train={len(train_flights)}  val={len(val_flights)}  test={len(test_flights)}")
    print(f"NEW base-trajectory split: "
          f"train_bases={len(train_bases)}  val_bases={len(val_bases)}  test_bases={len(test_bases)}")

    # Sanity check: confirm no base trajectory crosses splits under the new scheme
    new_split_of = {f: "train" for f in train_flights}
    new_split_of.update({f: "val" for f in val_flights})
    new_split_of.update({f: "test" for f in test_flights})
    crossing = 0
    for b in unique_bases:
        members = [f for f in unique_flights if base_of(f) == b]
        splits_hit = set(new_split_of[f] for f in members)
        if len(splits_hit) > 1:
            crossing += 1
    print(f"Base trajectories crossing splits under NEW scheme: {crossing}  (should be 0)")
    assert crossing == 0, "New split still leaks -- do not proceed until this is 0."

    # Compare how many flights changed split vs the old (leaky) assignment
    moved = sum(1 for f in unique_flights if old_split_of[f] != new_split_of[f])
    print(f"\nFlights whose split assignment changed old -> new: {moved} / {len(unique_flights)}")

    # --- Build corrected X/y arrays for val, and re-run coverage check ---
    val_mask = np.isin(flight_ids, val_flights)
    X_val, y_val = X[val_mask], y[val_mask]
    print(f"\nCorrected val set: {len(X_val)} windows from {len(val_flights)} flights "
          f"({len(val_bases)} unique base trajectories, all disjoint from train/test)")

    print("\nLoading ensemble + calibration params (unchanged from training -- note these")
    print("were themselves FIT on the old, leaky train split, so even this corrected-val")
    print("check is only a first honest read, not a final answer -- see printed guidance below) ...")
    models, scaler, dropped_idx_list = load_ensemble()
    params = load_calibration_params()

    print("\n=== Per-bucket coverage on CORRECTED (leakage-free) val split ===")
    results = check_per_bucket_coverage(X_val, y_val, models, scaler, dropped_idx_list, params)
    print(results)

    print("\n" + "=" * 78)
    print("WHAT THIS DOES AND DOES NOT TELL YOU")
    print("=" * 78)
    print("This corrected-val run is the first trustworthy read of coverage since the")
    print("leakage was introduced, but it is NOT the final answer, because:")
    print("  - The ensemble in train_rul_ensemble.py's saved weights was TRAINED on the")
    print("    OLD leaky train split, which included near-duplicate copies of some of")
    print("    the flights now correctly held out in val/test above. So the model may")
    print("    still be getting an unfair advantage on this val read specifically for")
    print("    flights whose base trajectory happened to stay in train under both splits.")
    print("  - Full correctness requires RETRAINING the ensemble from scratch using the")
    print("    NEW base-trajectory-grouped split for train too, then recalibrating")
    print("    (fit_calibration) on the new val predictions, THEN re-checking coverage.")
    print("  - Only after that retrain+recal should run_coverage_check.py ever be pointed")
    print("    at the NEW test split -- and note test itself changed composition (some")
    print("    flights moved in/out of test vs the original leaky partition), so the")
    print("    original 86.5% number is not comparable to whatever comes out next.")
    print("\nNext step: wire this same base_trajectory_id()-grouped split into the actual")
    print("training script (train_rul_ensemble.py) and calibration fitting, retrain the")
    print("full ensemble, refit calibration, THEN re-run this corrected-val check again")
    print("before ever touching test.")


if __name__ == "__main__":
    main()