"""
pod_sweep.py -- probability-of-detection (POD) curves per fault type,
as a function of fault severity and sensor noise.

Detection rule (stated so the curves are interpretable):
  A flight counts as DETECTED if, among windows whose end is at/after fault
  onset, the classifier raises K consecutive non-'none' predictions.
  "Correct-class" detection additionally requires those K predictions to be
  the true fault class.
  A healthy flight counts as a FALSE ALARM if K consecutive non-'none'
  predictions occur anywhere in it.

a90   = smallest severity on the grid with detection rate >= 0.90
a90/95 = smallest severity whose one-sided 95% lower confidence bound
         (Clopper-Pearson) on the detection rate is >= 0.90
         (with 30 flights this requires 30/30 detected).

Run from the project root:
    python -u pod_sweep.py          # full run (30 flights per cell)
    python -u pod_sweep.py 10       # quick run (10 flights per cell)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta
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
OUT_JSON = ROOT / "docs" / "pod_results.json"
OUT_CSV = ROOT / "docs" / "pod_results.csv"

CLASS_NAMES = ["cooling_degradation", "misfire", "none", "oil_issue",
               "overheat", "sensor_drift", "vibration_fault"]
NONE_IDX = CLASS_NAMES.index("none")

SEVERITIES = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
NOISE_LEVELS = (0.0, 0.02, 0.05)      # fraction of each sensor's healthy span
ONSET_FRAC = 0.4
DURATION = 300
K_CONSEC = 3
BASE_SEED = 900_000                   # high offset, to stay clear of training seeds
DROP_COLUMNS = ["label", "flight_id", "window_start", "window_end", "base_id"]

N_FLIGHTS = int(sys.argv[1]) if len(sys.argv) > 1 else 30


def span(sensor):
    lo, hi = HEALTHY_RANGES[sensor]
    return hi - lo


def add_noise(df, level, rng):
    d = df.copy()
    if level > 0:
        for s in SENSOR_FIELDS:
            d[s] = d[s] + rng.normal(0.0, level * span(s), size=len(d))
    return d


def first_run_end(flags, k):
    """Index of the last element of the first run of k consecutive True, else None."""
    run = 0
    for i, f in enumerate(flags):
        run = run + 1 if f else 0
        if run >= k:
            return i
    return None


def score_flight(df, fault, model, ref_cols):
    """Returns (detected_any, detected_correct, delay_timesteps or None)."""
    feats = process_flight_df(df, flight_id_override="pod")
    if len(feats) == 0:
        return False, False, None
    pred = model.predict(feats[ref_cols])
    ends = feats["window_end"].to_numpy()

    if fault == "none":
        end_i = first_run_end(pred != NONE_IDX, K_CONSEC)
        return end_i is not None, False, None

    onset = int(df["fault_onset_idx"].iloc[0]) if "fault_onset_idx" in df.columns \
        else int(len(df) * ONSET_FRAC)
    after = ends >= onset
    pred_after = pred[after]
    ends_after = ends[after]
    true_idx = CLASS_NAMES.index(fault)

    i_any = first_run_end(pred_after != NONE_IDX, K_CONSEC)
    i_cor = first_run_end(pred_after == true_idx, K_CONSEC)
    delay = int(ends_after[i_any] - onset) if i_any is not None else None
    return i_any is not None, i_cor is not None, delay


def lower_bound(k, n, conf=0.95):
    if k == 0:
        return 0.0
    return float(beta.ppf(1 - conf, k, n - k + 1))


def main():
    ref_cols = [c for c in pd.read_csv(FEATURES_CSV, nrows=1).columns if c not in DROP_COLUMNS]
    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))

    faults = [f for f in INJECTORS if f in CLASS_NAMES and f != "none"]
    print(f"Faults: {faults}")
    print(f"Flights per cell: {N_FLIGHTS}, severities: {SEVERITIES}, noise: {NOISE_LEVELS}")

    bases = [simulate_healthy_flight(duration_timesteps=DURATION, seed=BASE_SEED + i)
             for i in range(N_FLIGHTS)]
    for b in bases:
        b["fault_type"] = "none"
        b["flight_id"] = "pod"

    records = []
    t0 = time.time()

    # false alarms on healthy flights, per noise level
    for noise in NOISE_LEVELS:
        rng = np.random.default_rng(BASE_SEED + 7)
        alarms = 0
        for b in bases:
            d = add_noise(b, noise, rng)
            hit, _, _ = score_flight(d, "none", model, ref_cols)
            alarms += int(hit)
        records.append({"fault": "none", "noise": noise, "severity": 0.0,
                        "n": N_FLIGHTS, "detected": alarms, "correct": 0,
                        "median_delay": None, "failed": 0})
        print(f"[healthy] noise={noise:.0%}: false-alarm flights {alarms}/{N_FLIGHTS} "
              f"({time.time() - t0:.0f}s)", flush=True)

    for fault in faults:
        for noise in NOISE_LEVELS:
            for sev in SEVERITIES:
                rng = np.random.default_rng(BASE_SEED + 13)
                det = cor = failed = 0
                delays = []
                for i, b in enumerate(bases):
                    try:
                        inj = INJECTORS[fault](b.copy(), onset_frac=ONSET_FRAC,
                                               severity=sev, seed=BASE_SEED + 100 + i)
                        onset = int(inj["fault_onset_idx"].iloc[0]) if "fault_onset_idx" in inj.columns \
                            else int(len(inj) * ONSET_FRAC)
                        inj["fault_type"] = np.where(np.arange(len(inj)) >= onset, fault, "none")
                        inj["flight_id"] = "pod"
                        d = add_noise(inj, noise, rng)
                        a, c, dl = score_flight(d, fault, model, ref_cols)
                    except Exception as e:      # keep a long run alive, but count it
                        failed += 1
                        continue
                    det += int(a)
                    cor += int(c)
                    if dl is not None:
                        delays.append(dl)
                records.append({"fault": fault, "noise": noise, "severity": sev,
                                "n": N_FLIGHTS - failed, "detected": det, "correct": cor,
                                "median_delay": float(np.median(delays)) if delays else None,
                                "failed": failed})
            print(f"[{fault}] noise={noise:.0%} done ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(records)
    df["rate"] = df["detected"] / df["n"].clip(lower=1)
    df["rate_correct"] = df["correct"] / df["n"].clip(lower=1)
    df["lcb95"] = [lower_bound(k, n) for k, n in zip(df["detected"], df["n"])]

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    with open(OUT_JSON, "w") as f:
        json.dump(json.loads(df.to_json(orient="records")), f, indent=2)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 70)
    print("FALSE-ALARM RATE ON HEALTHY FLIGHTS (per flight)")
    print("=" * 70)
    h = df[df["fault"] == "none"][["noise", "detected", "n"]]
    print(h.to_string(index=False))

    for fault in faults:
        sub = df[df["fault"] == fault]
        print("\n" + "=" * 70)
        print(f"{fault}: detection rate (any-fault alarm)")
        print("=" * 70)
        print(sub.pivot(index="severity", columns="noise", values="rate")
                 .round(2).to_string())
        print(f"\n{fault}: correct-class detection rate")
        print(sub.pivot(index="severity", columns="noise", values="rate_correct")
                 .round(2).to_string())
        print(f"\n{fault}: median detection delay after onset (timesteps)")
        print(sub.pivot(index="severity", columns="noise", values="median_delay")
                 .round(0).to_string())

    print("\n" + "=" * 70)
    print("SUMMARY: smallest severity detected reliably")
    print("=" * 70)
    rows = []
    for fault in faults:
        for noise in NOISE_LEVELS:
            sub = df[(df["fault"] == fault) & (df["noise"] == noise)].sort_values("severity")
            a90 = sub[sub["rate"] >= 0.90]["severity"]
            a9095 = sub[sub["lcb95"] >= 0.90]["severity"]
            rows.append({"fault": fault, "noise": f"{noise:.0%}",
                         "a90": float(a90.iloc[0]) if len(a90) else "not reached",
                         "a90/95": float(a9095.iloc[0]) if len(a9095) else "not reached"})
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nSaved {OUT_CSV} and {OUT_JSON}")


if __name__ == "__main__":
    main()