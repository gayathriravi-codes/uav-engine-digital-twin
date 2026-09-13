"""
test_whatif_all_faults.py

Sanity check for run_whatif() across ALL fault types, not just
cooling_degradation. Run this from the project root:

    python test_whatif_all_faults.py

For each fault type, it:
  1. Picks one sample flight from data/raw/
  2. Reads fault_type, fault_onset_idx, and flight length straight from
     the CSV (same convention run_whatif() expects)
  3. Sets window_start = 70% of the way from onset to end-of-flight
     (same placement as the original single-fault demo)
  4. Runs run_whatif() and prints point estimates + deltas per scenario
  5. Checks whether deltas are monotonically ordered by severity, and
     flags non-monotonic results

IMPORTANT: misfire is EXPECTED to be non-monotonic. It's injected as a
probabilistic event (random roll per timestep) rather than a smooth
ramp, so severity doesn't map cleanly onto a single ordered outcome.
That's a real, explainable property of the fault model -- not a bug.
This script labels it as such rather than flagging it as a failure.
"""

import glob
import os
import sys

import pandas as pd

# Make sure the project root is on the path when run from repo root
sys.path.insert(0, os.getcwd())

from models.rul.whatif_engine import run_whatif  # noqa: E402

DATA_DIR = os.path.join("data", "raw")

# Fault types expected in the dataset (5 fault types + healthy is not
# applicable here since what-if only makes sense mid-fault)
FAULT_TYPES = [
    "misfire",
    "overheat",
    "cooling_degradation",
    "oil_issue",
    "sensor_drift",
    "vibration_fault",
]

# Known non-monotonic fault types and WHY -- extend this if other faults
# turn out to have similar probabilistic/non-ramped injection logic.
KNOWN_NON_MONOTONIC = {
    "misfire": (
        "misfire is injected as a probabilistic event (random roll per "
        "timestep), not a smooth ramp -- severity doesn't map cleanly "
        "onto a single ordered outcome. Expected, not a bug."
    ),
}


def find_sample_flight(fault_type: str):
    """Find one sample CSV for a given fault type."""
    pattern = os.path.join(DATA_DIR, f"{fault_type}_*.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return matches[0]


def check_differentiated(results, tol=1e-6):
    """
    NOTE: the 4 scenarios are DIFFERENT interventions (reduce throttle,
    adjust cooling, reduce RPM), not a single dial turned up in severity --
    so there's no reason to expect them ordered relative to EACH OTHER.
    A cooling fix should help a cooling fault more than a vibration fix
    would, and vice versa. Checking cross-intervention ordering was the
    original version of this script's bug.

    What actually indicates a healthy result:
      1. baseline ("continue_current") has the lowest point estimate
      2. every intervention's delta differs meaningfully from 0.0
         (a delta of ~0 for ALL interventions means severity isn't
         differentiating anything -- the real bug symptom, as seen with
         the vibration/vibration_fault key mismatch)
    """
    baseline = next((r for r in results if r["scenario"] == "continue_current"), results[0])
    interventions = [r for r in results if r is not baseline]

    baseline_is_lowest = all(
        baseline["point_estimate_minutes"] <= r["point_estimate_minutes"]
        for r in interventions
    )
    all_flat = all(abs(r["delta_minutes"]) < tol for r in interventions)

    return baseline_is_lowest, all_flat


def main():
    summary = []

    for fault_type in FAULT_TYPES:
        print("=" * 70)
        print(f"FAULT TYPE: {fault_type}")
        print("=" * 70)

        flight_path = find_sample_flight(fault_type)
        if flight_path is None:
            print(f"  [SKIPPED] No sample flight found for '{fault_type}' "
                  f"in {DATA_DIR}. Run fault_injectors.py to regenerate "
                  f"data/raw/ if this is unexpected.\n")
            summary.append((fault_type, "NO DATA", None))
            continue

        df = pd.read_csv(flight_path)

        if "fault_type" not in df.columns or "fault_onset_idx" not in df.columns:
            print(f"  [SKIPPED] {flight_path} is missing expected columns "
                  f"(fault_type, fault_onset_idx). Check schema.\n")
            summary.append((fault_type, "BAD SCHEMA", None))
            continue

        onset_idx = int(df["fault_onset_idx"].iloc[0])
        duration = len(df)
        window_start = onset_idx + int(0.70 * (duration - onset_idx))

        print(f"  Flight: {os.path.basename(flight_path)}")
        print(f"  onset_idx={onset_idx}  duration={duration}  "
              f"window_start={window_start}\n")

        try:
            results = run_whatif(
                fault_type=fault_type,
                onset_idx=onset_idx,
                duration=duration,
                window_start=window_start,
            )
        except Exception as e:
            print(f"  [ERROR] run_whatif() raised: {e}\n")
            summary.append((fault_type, "ERROR", str(e)))
            continue

        for r in results:
            print(f"  {r['label']:<38} point_est={r['point_estimate_minutes']:>8.1f}"
                  f"  delta={r['delta_minutes']:>+8.1f}")

        baseline_is_lowest, all_flat = check_differentiated(results)

        if all_flat:
            verdict = ("FLAT (BUG-LIKE) -- all interventions returned ~0 "
                       "delta vs baseline. Severity isn't differentiating "
                       "anything for this fault type. Check the fault_type "
                       "string matches the real INJECTORS dict key exactly "
                       "(this is what caused the vibration/vibration_fault "
                       "mismatch).")
            status = "FLAG"
        elif not baseline_is_lowest:
            verdict = ("UNEXPECTED -- baseline (continue_current) is not "
                       "the lowest point estimate. An intervention scoring "
                       "worse than doing nothing needs investigation.")
            status = "FLAG"
        elif fault_type in KNOWN_NON_MONOTONIC:
            verdict = f"OK (known characteristic: {KNOWN_NON_MONOTONIC[fault_type]})"
            status = "OK"
        else:
            verdict = "OK (baseline lowest, interventions differentiated)"
            status = "OK"

        print(f"\n  Verdict: {verdict}\n")
        summary.append((fault_type, status, verdict))

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for fault_type, status, verdict in summary:
        print(f"  {fault_type:<22} {status:<10} {verdict or ''}")


if __name__ == "__main__":
    main()