"""
pod_followup.py -- two follow-ups to pod_sweep.py.

Imports helpers from pod_sweep.py, so keep pod_sweep.py unchanged and in the
same folder. Run from the project root with NO arguments:
    python -u pod_followup.py        (about 10 minutes)

Part 1: per-flight false-alarm rate on many more healthy flights (300, seeds
        disjoint from the 30 used in pod_sweep.py), at each noise level, with
        two-sided Clopper-Pearson 95% intervals. Also reports the per-window
        false-alarm rate (comparable to the earlier ~3% figure) and which class
        the false alarms hallucinate.

Part 2: sensor_drift detection split by sub_type (bias / drift / stuck) and
        target sensor (oil_pressure / vibration), noise-free, 30 flights per
        cell, same severity grid and detection rule as pod_sweep.py.

Outputs: docs/pod_healthy_fa.json, docs/pod_drift_strata.csv/.json
"""

import json
import time
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import beta
from xgboost import XGBClassifier

import pod_sweep as ps
from models.classification.feature_extraction import process_flight_df
from simulator.simulate_healthy import simulate_healthy_flight
from simulator.fault_injectors import INJECTORS

N_HEALTHY = 300
HEALTHY_SEED0 = ps.BASE_SEED + 5000
N_DRIFT = 30
DRIFT_NOISE = 0.0
SUB_TYPES = ("bias", "drift", "stuck")
TARGETS = ("oil_pressure", "vibration")

OUT_HEALTHY = ps.ROOT / "docs" / "pod_healthy_fa.json"
OUT_DRIFT_CSV = ps.ROOT / "docs" / "pod_drift_strata.csv"
OUT_DRIFT_JSON = ps.ROOT / "docs" / "pod_drift_strata.json"


def cp_interval(k, n, conf=0.95):
    """Two-sided Clopper-Pearson interval."""
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def healthy_part(model, ref_cols, t0):
    bases = []
    for i in range(N_HEALTHY):
        b = simulate_healthy_flight(duration_timesteps=ps.DURATION, seed=HEALTHY_SEED0 + i)
        b["fault_type"] = "none"
        b["flight_id"] = "pod"
        bases.append(b)
    print(f"[healthy] generated {N_HEALTHY} flights ({time.time() - t0:.0f}s)", flush=True)

    results = []
    for noise in ps.NOISE_LEVELS:
        rng = np.random.default_rng(ps.BASE_SEED + 7)
        alarms = scored = win_total = win_bad = 0
        hallucinated = Counter()
        for b in bases:
            d = ps.add_noise(b, noise, rng)
            feats = process_flight_df(d, flight_id_override="pod")
            if len(feats) == 0:
                continue
            scored += 1
            pred = model.predict(feats[ref_cols])
            bad = pred != ps.NONE_IDX
            win_total += len(pred)
            win_bad += int(bad.sum())
            end_i = ps.first_run_end(bad, ps.K_CONSEC)
            if end_i is not None:
                alarms += 1
                hallucinated[ps.CLASS_NAMES[int(pred[end_i])]] += 1
        lo, hi = cp_interval(alarms, scored)
        rec = {"noise": noise, "flights": scored, "alarm_flights": alarms,
               "flight_rate": alarms / max(scored, 1), "ci95_low": lo, "ci95_high": hi,
               "windows": win_total, "alarm_windows": win_bad,
               "window_rate": win_bad / max(win_total, 1),
               "alarm_class_counts": dict(hallucinated)}
        results.append(rec)
        print(f"[healthy] noise={noise:.0%}: {alarms}/{scored} flights alarm "
              f"({rec['flight_rate']:.1%}, 95% CI {lo:.1%}-{hi:.1%}); "
              f"per-window {rec['window_rate']:.1%}; classes {dict(hallucinated)} "
              f"({time.time() - t0:.0f}s)", flush=True)

    ps.OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_HEALTHY, "w") as f:
        json.dump(results, f, indent=2)
    return results


def drift_part(model, ref_cols, t0):
    bases = [simulate_healthy_flight(duration_timesteps=ps.DURATION, seed=ps.BASE_SEED + i)
             for i in range(N_DRIFT)]
    for b in bases:
        b["fault_type"] = "none"
        b["flight_id"] = "pod"

    records = []
    first_err = None
    for st in SUB_TYPES:
        for ts in TARGETS:
            for sev in ps.SEVERITIES:
                rng = np.random.default_rng(ps.BASE_SEED + 13)
                det = cor = failed = 0
                delays = []
                for i, b in enumerate(bases):
                    try:
                        inj = INJECTORS["sensor_drift"](
                            b.copy(), onset_frac=ps.ONSET_FRAC, severity=sev,
                            seed=ps.BASE_SEED + 100 + i, sub_type=st, target_sensor=ts)
                        onset = int(inj["fault_onset_idx"].iloc[0]) if "fault_onset_idx" in inj.columns \
                            else int(len(inj) * ps.ONSET_FRAC)
                        inj["fault_type"] = np.where(np.arange(len(inj)) >= onset,
                                                     "sensor_drift", "none")
                        inj["flight_id"] = "pod"
                        d = ps.add_noise(inj, DRIFT_NOISE, rng)
                        a, c, dl = ps.score_flight(d, "sensor_drift", model, ref_cols)
                    except Exception as e:
                        failed += 1
                        if first_err is None:
                            first_err = repr(e)
                        continue
                    det += int(a)
                    cor += int(c)
                    if dl is not None:
                        delays.append(dl)
                n = N_DRIFT - failed
                records.append({"sub_type": st, "target": ts, "severity": sev, "n": n,
                                "detected": det, "correct": cor, "failed": failed,
                                "median_delay": float(np.median(delays)) if delays else None})
            print(f"[sensor_drift] {st}/{ts} done ({time.time() - t0:.0f}s)", flush=True)

    if first_err:
        print(f"\nNOTE: some injections failed; first error: {first_err}")

    df = pd.DataFrame(records)
    df["rate"] = df["detected"] / df["n"].clip(lower=1)
    df["rate_correct"] = df["correct"] / df["n"].clip(lower=1)
    df.to_csv(OUT_DRIFT_CSV, index=False)
    with open(OUT_DRIFT_JSON, "w") as f:
        json.dump(json.loads(df.to_json(orient="records")), f, indent=2)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 70)
    print("sensor_drift detection rate by sub_type (both targets pooled)")
    print("=" * 70)
    g = df.groupby(["sub_type", "severity"])[["detected", "n"]].sum().reset_index()
    g["rate"] = g["detected"] / g["n"].clip(lower=1)
    print(g.pivot(index="severity", columns="sub_type", values="rate").round(2).to_string())

    print("\n" + "=" * 70)
    print("sensor_drift detection rate by sub_type x target")
    print("=" * 70)
    print(df.pivot(index="severity", columns=["sub_type", "target"], values="rate")
            .round(2).to_string())

    print("\n" + "=" * 70)
    print("sensor_drift correct-class rate by sub_type x target")
    print("=" * 70)
    print(df.pivot(index="severity", columns=["sub_type", "target"], values="rate_correct")
            .round(2).to_string())

    print("\n" + "=" * 70)
    print("sensor_drift median detection delay (timesteps) by sub_type x target")
    print("=" * 70)
    print(df.pivot(index="severity", columns=["sub_type", "target"], values="median_delay")
            .round(0).to_string())
    return df


def main():
    ref_cols = [c for c in pd.read_csv(ps.FEATURES_CSV, nrows=1).columns
                if c not in ps.DROP_COLUMNS]
    model = XGBClassifier()
    model.load_model(str(ps.MODEL_PATH))
    t0 = time.time()

    healthy_part(model, ref_cols, t0)
    drift_part(model, ref_cols, t0)

    print(f"\nSaved {OUT_HEALTHY}, {OUT_DRIFT_CSV} and {OUT_DRIFT_JSON}")


if __name__ == "__main__":
    main()