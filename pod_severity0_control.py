"""
pod_severity0_control.py -- is low-severity "detection" a noise cue, not a fault?

Why this exists
---------------
The fault injectors add extra Gaussian noise on every row after onset
(overheat, cooling, oil) and a vibration burstiness sigma = 0.1 + 0.4*d that
does not scale with severity. Misfire also keeps a base event probability of
0.05 at severity 0, and a stuck sensor ignores severity entirely. So even at
severity 0 a "faulted" flight can look different from a healthy one, and the
classifier leans on window standard deviations.

This script runs the SAME detection rule as pod_sweep.py (K = 3 consecutive
non-'none' windows after onset) at severity 0.0 and compares it with:
  (a) severity 0.1  -- the smallest severity in the POD grid
  (b) healthy flights -- the false-alarm rate (K consecutive non-'none' windows
      anywhere in a healthy flight; same rule, same seeds, same noise)

Reading the result
------------------
If the severity-0 detection rate is well above the healthy false-alarm rate,
the detector is responding to severity-independent noise, and the a90 = 0.1
values in the checklist should not be presented as detection limits for the
fault offset. If severity 0 is close to the healthy rate, they can stand.

Run from the project root (confirm pwd first):
    python -u pod_severity0_control.py          # 30 flights per cell
    python -u pod_severity0_control.py 10       # quick run

Output: docs/pod_severity0_control.csv and a printed table.
Nothing in the existing project is modified.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "simulator"))

from schema import SENSOR_FIELDS, HEALTHY_RANGES
from models.classification.feature_extraction import process_flight_df
from simulator.simulate_healthy import simulate_healthy_flight
from simulator.fault_injectors import INJECTORS

MODEL_PATH = ROOT / "models" / "classification" / "xgboost_fault_classifier_aug.json"
FEATURES_CSV = ROOT / "data" / "processed" / "classification_features.csv"
OUT_CSV = ROOT / "docs" / "pod_severity0_control.csv"

CLASS_NAMES = ["cooling_degradation", "misfire", "none", "oil_issue",
               "overheat", "sensor_drift", "vibration_fault"]
NONE_IDX = CLASS_NAMES.index("none")
SEVERITIES = (0.0, 0.1)
NOISE_LEVELS = (0.0, 0.02, 0.05)     # fraction of FULL range, added on top of default noise
ONSET_FRAC = 0.4
DURATION = 300
K_CONSEC = 3
BASE_SEED = 900_000                  # same base as pod_sweep.py, so flights match
DROP_COLUMNS = ["label", "flight_id", "window_start", "window_end", "base_id"]
N_FLIGHTS = int(sys.argv[1]) if len(sys.argv) > 1 else 30


def full_range(sensor):
    lo, hi = HEALTHY_RANGES[sensor]
    return hi - lo


def add_noise(df, level, rng):
    d = df.copy()
    if level > 0:
        for s in SENSOR_FIELDS:
            d[s] = d[s] + rng.normal(0.0, level * full_range(s), size=len(d))
    return d


def first_run_end(flags, k):
    run = 0
    for i, f in enumerate(flags):
        run = run + 1 if f else 0
        if run >= k:
            return i
    return None


def detected(df, is_healthy, model, ref_cols):
    feats = process_flight_df(df, flight_id_override="ctl")
    if len(feats) == 0:
        return False
    pred = model.predict(feats[ref_cols])
    if is_healthy:
        return first_run_end(pred != NONE_IDX, K_CONSEC) is not None
    onset = int(df["fault_onset_idx"].iloc[0]) if "fault_onset_idx" in df.columns \
        else int(len(df) * ONSET_FRAC)
    after = feats["window_end"].to_numpy() >= onset
    return first_run_end(pred[after] != NONE_IDX, K_CONSEC) is not None


def main():
    ref_cols = [c for c in pd.read_csv(FEATURES_CSV, nrows=1).columns if c not in DROP_COLUMNS]
    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))

    faults = [f for f in INJECTORS if f in CLASS_NAMES and f != "none"]
    bases = []
    for i in range(N_FLIGHTS):
        b = simulate_healthy_flight(duration_timesteps=DURATION, seed=BASE_SEED + i)
        b["fault_type"] = "none"
        b["flight_id"] = "ctl"
        bases.append(b)

    rows, t0 = [], time.time()

    for noise in NOISE_LEVELS:
        rng = np.random.default_rng(BASE_SEED + 7)
        k = sum(int(detected(add_noise(b, noise, rng), True, model, ref_cols)) for b in bases)
        rows.append({"fault": "healthy (false alarm)", "severity": "-", "noise": noise,
                     "n": N_FLIGHTS, "detected": k, "failed": 0})
        print(f"[healthy] noise={noise:.0%}: {k}/{N_FLIGHTS} ({time.time() - t0:.0f}s)", flush=True)

    for fault in faults:
        for sev in SEVERITIES:
            for noise in NOISE_LEVELS:
                rng = np.random.default_rng(BASE_SEED + 13)
                k = failed = 0
                for i, b in enumerate(bases):
                    try:
                        inj = INJECTORS[fault](b.copy(), onset_frac=ONSET_FRAC,
                                               severity=sev, seed=BASE_SEED + 100 + i)
                        onset = int(inj["fault_onset_idx"].iloc[0]) if "fault_onset_idx" in inj.columns \
                            else int(len(inj) * ONSET_FRAC)
                        inj["fault_type"] = np.where(np.arange(len(inj)) >= onset, fault, "none")
                        inj["flight_id"] = "ctl"
                        k += int(detected(add_noise(inj, noise, rng), False, model, ref_cols))
                    except Exception as e:
                        failed += 1
                rows.append({"fault": fault, "severity": str(sev), "noise": noise,
                             "n": N_FLIGHTS - failed, "detected": k, "failed": failed})
            print(f"[{fault}] severity={sev} done ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df["rate"] = (df["detected"] / df["n"].clip(lower=1)).round(2)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 70)
    print("Detection rate at severity 0.0 vs 0.1 vs healthy false-alarm rate")
    print("(rows: fault/severity, columns: noise as fraction of full range)")
    print("=" * 70)
    print(df.pivot_table(index=["fault", "severity"], columns="noise", values="rate",
                         aggfunc="first").to_string())
    print(f"\nSaved {OUT_CSV}")
    print("If severity 0.0 is well above the healthy row, low-severity detection "
          "is partly a noise cue.")


if __name__ == "__main__":
    main()