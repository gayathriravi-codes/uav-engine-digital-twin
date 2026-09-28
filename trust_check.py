"""
trust_check.py -- does sensor_trust.py catch the sensor_drift cases that the
classifier misses (especially a stuck oil_pressure sensor)?

Keep pod_sweep.py, pod_followup.py and sensor_trust.py in the project root.
Run from the project root with NO arguments:
    python -u trust_check.py        (roughly 8-10 minutes)

Method (mirrors pod_sweep.py so the numbers are comparable):
  * Windows of WINDOW_SIZE rows, stride STRIDE (from schema.py), each passed to
    evaluate_sensor_trust(). A window's end is start + WINDOW_SIZE - 1, which may
    differ by one timestep from the classifier's window_end convention.
  * DETECTED (target) = the injected sensor is listed untrusted in K=3
    consecutive windows whose end is at/after fault onset.
  * DETECTED (any)    = any sensor is untrusted in K=3 consecutive such windows
    (weaker: can be triggered by a wrong sensor).
  * Delay = end of the window completing the first K-run minus onset.

Part 1: healthy flights (300, same seeds as pod_followup.py) at 0/2/5% noise:
        per-flight and per-window false-flag rates, and which sensor:check
        combinations fire.
Part 2: sensor_drift by sub_type x target x severity, noise-free, 30 flights
        per cell, same flights and seeds as pod_followup.py.
Part 3: stuck sensor with noise present (2%, 5% of span). In pod_sweep.py noise
        was added AFTER injection, which puts noise on a frozen sensor; here
        noise is added to the healthy flight BEFORE injection, so the stuck
        sensor stays frozen at its onset value, as a real one would.

Outputs: docs/trust_check_healthy.json, docs/trust_check_drift.csv/.json,
         docs/trust_check_stuck_noise.json
"""

import json
import time
from collections import Counter

import numpy as np
import pandas as pd

import pod_sweep as ps
from pod_followup import cp_interval
from schema import WINDOW_SIZE, STRIDE
from sensor_trust import evaluate_sensor_trust
from simulator.simulate_healthy import simulate_healthy_flight
from simulator.fault_injectors import INJECTORS

N_HEALTHY = 300
HEALTHY_SEED0 = ps.BASE_SEED + 5000
N_DRIFT = 30
SUB_TYPES = ("bias", "drift", "stuck")
TARGETS = ("oil_pressure", "vibration")
STUCK_NOISE_LEVELS = (0.02, 0.05)

OUT_HEALTHY = ps.ROOT / "docs" / "trust_check_healthy.json"
OUT_DRIFT_CSV = ps.ROOT / "docs" / "trust_check_drift.csv"
OUT_DRIFT_JSON = ps.ROOT / "docs" / "trust_check_drift.json"
OUT_STUCK_NOISE = ps.ROOT / "docs" / "trust_check_stuck_noise.json"


def trust_windows(df):
    """[(window_end_index, evaluate_sensor_trust result), ...] over the flight."""
    out = []
    for start in range(0, len(df) - WINDOW_SIZE + 1, STRIDE):
        w = df.iloc[start:start + WINDOW_SIZE]
        out.append((start + WINDOW_SIZE - 1, evaluate_sensor_trust(w)))
    return out


def score_trust(df, target, onset):
    """{'target': (hit, delay, checks), 'any': (hit, delay, checks)}"""
    wins = trust_windows(df)
    ends = np.array([e for e, _ in wins])
    idx_after = np.flatnonzero(ends >= onset)
    tests = (("target", lambda r: target in r["untrusted_sensors"]),
             ("any", lambda r: r["any_untrusted"]))
    out = {}
    for name, fn in tests:
        flags = np.array([fn(wins[j][1]) for j in idx_after], dtype=bool)
        i = ps.first_run_end(flags, ps.K_CONSEC) if len(flags) else None
        if i is None:
            out[name] = (False, None, [])
        else:
            j = idx_after[i]
            out[name] = (True, int(ends[j] - onset), wins[j][1]["flags"].get(target, []))
    return out


def make_bases(n, seed0):
    bases = []
    for i in range(n):
        b = simulate_healthy_flight(duration_timesteps=ps.DURATION, seed=seed0 + i)
        b["fault_type"] = "none"
        b["flight_id"] = "pod"
        bases.append(b)
    return bases


def inject_stuck_or_drift(b, i, sev, st, ts):
    inj = INJECTORS["sensor_drift"](
        b.copy(), onset_frac=ps.ONSET_FRAC, severity=sev,
        seed=ps.BASE_SEED + 100 + i, sub_type=st, target_sensor=ts)
    onset = int(inj["fault_onset_idx"].iloc[0]) if "fault_onset_idx" in inj.columns \
        else int(len(inj) * ps.ONSET_FRAC)
    inj["fault_type"] = np.where(np.arange(len(inj)) >= onset, "sensor_drift", "none")
    inj["flight_id"] = "pod"
    return inj, onset


def healthy_part(t0):
    bases = make_bases(N_HEALTHY, HEALTHY_SEED0)
    results = []
    for noise in ps.NOISE_LEVELS:
        rng = np.random.default_rng(ps.BASE_SEED + 7)
        alarm_flights = win_total = win_bad = 0
        checks = Counter()
        for b in bases:
            d = ps.add_noise(b, noise, rng)
            wins = trust_windows(d)
            flags = np.array([r["any_untrusted"] for _, r in wins], dtype=bool)
            win_total += len(flags)
            win_bad += int(flags.sum())
            i = ps.first_run_end(flags, ps.K_CONSEC)
            if i is not None:
                alarm_flights += 1
                for s, cl in wins[i][1]["flags"].items():
                    for c in cl:
                        checks[f"{s}:{c}"] += 1
        lo, hi = cp_interval(alarm_flights, N_HEALTHY)
        rec = {"noise": noise, "flights": N_HEALTHY, "alarm_flights": alarm_flights,
               "flight_rate": alarm_flights / N_HEALTHY, "ci95_low": lo, "ci95_high": hi,
               "windows": win_total, "alarm_windows": win_bad,
               "window_rate": win_bad / max(win_total, 1),
               "checks_at_alarm": dict(checks.most_common(8))}
        results.append(rec)
        print(f"[healthy] noise={noise:.0%}: {alarm_flights}/{N_HEALTHY} flights untrusted "
              f"({rec['flight_rate']:.1%}, 95% CI {lo:.1%}-{hi:.1%}); "
              f"per-window {rec['window_rate']:.1%}; top {dict(checks.most_common(4))} "
              f"({time.time() - t0:.0f}s)", flush=True)
    OUT_HEALTHY.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_HEALTHY, "w") as f:
        json.dump(results, f, indent=2)


def drift_part(t0):
    bases = make_bases(N_DRIFT, ps.BASE_SEED)
    records = []
    first_err = None
    for st in SUB_TYPES:
        for ts in TARGETS:
            for sev in ps.SEVERITIES:
                det_t = det_a = failed = 0
                delays = []
                checks = Counter()
                for i, b in enumerate(bases):
                    try:
                        inj, onset = inject_stuck_or_drift(b, i, sev, st, ts)
                        res = score_trust(inj, ts, onset)
                    except Exception as e:
                        failed += 1
                        if first_err is None:
                            first_err = repr(e)
                        continue
                    hit_t, delay_t, chk = res["target"]
                    det_t += int(hit_t)
                    det_a += int(res["any"][0])
                    if hit_t:
                        delays.append(delay_t)
                        for c in chk:
                            checks[c] += 1
                n = N_DRIFT - failed
                records.append({"sub_type": st, "target": ts, "severity": sev, "n": n,
                                "detected_target": det_t, "detected_any": det_a,
                                "failed": failed,
                                "median_delay": float(np.median(delays)) if delays else None,
                                "checks": json.dumps(dict(checks))})
            print(f"[sensor_drift] {st}/{ts} done ({time.time() - t0:.0f}s)", flush=True)

    if first_err:
        print(f"\nNOTE: some injections/scoring failed; first error: {first_err}")

    df = pd.DataFrame(records)
    df["rate_target"] = df["detected_target"] / df["n"].clip(lower=1)
    df["rate_any"] = df["detected_any"] / df["n"].clip(lower=1)
    df.to_csv(OUT_DRIFT_CSV, index=False)
    with open(OUT_DRIFT_JSON, "w") as f:
        json.dump(json.loads(df.to_json(orient="records")), f, indent=2)

    pd.set_option("display.width", 200)
    for title, col in (("TARGET sensor flagged untrusted (detection rate)", "rate_target"),
                       ("ANY sensor flagged untrusted (detection rate)", "rate_any"),
                       ("median delay after onset, target flagged (timesteps)", "median_delay")):
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)
        p = df.pivot(index="severity", columns=["sub_type", "target"], values=col)
        print(p.round(2 if col != "median_delay" else 0).to_string())

    print("\n" + "=" * 70)
    print("which check flagged the target sensor (summed over severities)")
    print("=" * 70)
    for (st, ts), g in df.groupby(["sub_type", "target"]):
        tot = Counter()
        for s in g["checks"]:
            tot.update(json.loads(s))
        print(f"  {st}/{ts}: {dict(tot)}")


def stuck_noise_part(t0):
    bases = make_bases(N_DRIFT, ps.BASE_SEED)
    results = []
    for noise in STUCK_NOISE_LEVELS:
        for ts in TARGETS:
            rng = np.random.default_rng(ps.BASE_SEED + 21)
            det = failed = 0
            delays = []
            for i, b in enumerate(bases):
                try:
                    nb = ps.add_noise(b, noise, rng)      # noise BEFORE injection
                    inj, onset = inject_stuck_or_drift(nb, i, 1.0, "stuck", ts)
                    hit, delay, _ = score_trust(inj, ts, onset)["target"]
                except Exception:
                    failed += 1
                    continue
                det += int(hit)
                if hit:
                    delays.append(delay)
            n = N_DRIFT - failed
            rec = {"noise": noise, "target": ts, "n": n, "detected": det,
                   "rate": det / max(n, 1),
                   "median_delay": float(np.median(delays)) if delays else None}
            results.append(rec)
            print(f"[stuck, noise before injection] noise={noise:.0%} {ts}: "
                  f"{det}/{n} flagged, median delay {rec['median_delay']} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    with open(OUT_STUCK_NOISE, "w") as f:
        json.dump(results, f, indent=2)


def main():
    t0 = time.time()
    healthy_part(t0)
    drift_part(t0)
    print()
    stuck_noise_part(t0)
    print(f"\nSaved {OUT_HEALTHY}, {OUT_DRIFT_CSV}, {OUT_DRIFT_JSON} and {OUT_STUCK_NOISE}")


if __name__ == "__main__":
    main()