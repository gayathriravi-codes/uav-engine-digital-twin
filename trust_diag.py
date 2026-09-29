"""Diagnose sensor_trust sustained_mean_shift false alarms on healthy flights (0% noise)."""
import numpy as np
import pod_sweep as ps
from schema import WINDOW_SIZE, STRIDE
from sensor_trust import MEAN_SHIFT_CONFIG
from simulator.simulate_healthy import simulate_healthy_flight

N = 300
SEED0 = ps.BASE_SEED + 5000   # same healthy seeds as trust_check.py
K = 3
SENSORS = list(MEAN_SHIFT_CONFIG)


def stats(values, cfg):
    r, b = cfg["recent_window"], cfg["baseline_window"]
    if len(values) < r + b:
        return None
    base, rec = values[-(r + b):-r], values[-r:]
    sd = float(np.std(base))
    shift = abs(float(np.mean(rec)) - float(np.mean(base)))
    if sd < 1e-6:
        ratio = np.inf if shift > 1e-6 else 0.0
    else:
        ratio = shift / sd
    return ratio, shift, sd


def has_run(flags, k=K):
    run = 0
    for f in flags:
        run = run + 1 if f else 0
        if run >= k:
            return True
    return False


# per sensor: list (one entry per flight) of arrays [ratio, shift, sd] per window
data = {s: [] for s in SENSORS}
rpm_level = []
for i in range(N):
    df = simulate_healthy_flight(duration_timesteps=ps.DURATION, seed=SEED0 + i)
    rpm_level.append(float(df["rpm"].mean()))
    rows = {s: [] for s in SENSORS}
    for start in range(0, len(df) - WINDOW_SIZE + 1, STRIDE):
        w = df.iloc[start:start + WINDOW_SIZE]
        for s in SENSORS:
            st = stats(w[s].astype(float).values, MEAN_SHIFT_CONFIG[s])
            if st is not None:
                rows[s].append(st)
    for s in SENSORS:
        data[s].append(np.array(rows[s]))

print(f"WINDOW_SIZE={WINDOW_SIZE} STRIDE={STRIDE} flights={N}\n")
print("Reproduction of the current check (ratio > 3.0), compare trust_check_healthy.json:")
print("  expected: rpm 49 flights; egt 1; all-sensor window rate 11.3%")
tot_w = 0
for s in SENSORS:
    thr = MEAN_SHIFT_CONFIG[s]["threshold_std_multiples"]
    fl = sum(has_run(a[:, 0] > thr) for a in data[s] if len(a))
    nw = sum(int((a[:, 0] > thr).sum()) for a in data[s] if len(a))
    tw = sum(len(a) for a in data[s])
    tot_w = tw
    print(f"  {s:13s} flights with {K}-run: {fl:3d}/{N}   flagged windows: {nw}/{tw} ({nw / tw:.1%})")

s = "rpm"
thr = MEAN_SHIFT_CONFIG[s]["threshold_std_multiples"]
allw = np.vstack([a for a in data[s] if len(a)])
flag = allw[:, 0] > thr
print(f"\nrpm flagged windows: {int(flag.sum())} of {len(allw)}")
print(f"  median rpm level across flights: {np.median(rpm_level):.0f}")
print(f"  ratio in flagged windows  median {np.median(allw[flag, 0]):.2f}  max {np.max(allw[flag, 0]):.2f}")
print(f"  |shift| in flagged windows  median {np.median(allw[flag, 1]):.1f}  max {np.max(allw[flag, 1]):.1f}")
print(f"  baseline sd in flagged windows  median {np.median(allw[flag, 2]):.1f}")
print(f"  |shift| in ALL windows  median {np.median(allw[:, 1]):.1f}  p99 {np.quantile(allw[:, 1], .99):.1f}")
print(f"  baseline sd in ALL windows  median {np.median(allw[:, 2]):.1f}")

print("\nEffect of adding a minimum absolute shift (rpm units) on top of ratio > 3:")
floors = [0.0] + [float(np.quantile(allw[flag, 1], q)) for q in (0.25, 0.5, 0.75, 0.9)] \
         + [float(allw[flag, 1].max())]
lvl = float(np.median(rpm_level))
print("  floor(rpm)  floor(% of level)  flights_with_3run")
for fl_ in floors:
    n = sum(has_run((a[:, 0] > thr) & (a[:, 1] > fl_)) for a in data[s] if len(a))
    print(f"  {fl_:9.1f}   {100 * fl_ / lvl:8.2f}%          {n:3d}/{N}")