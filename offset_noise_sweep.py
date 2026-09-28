"""
offset_noise_sweep.py

Tests classifier robustness to sensor-level mismatch (not coupling):
  1. per-build calibration offsets on every sensor
  2. extra measurement noise on every sensor
Uses the existing generators and the same leakage-fixed classifier.

Run from the project root:
    python -u offset_noise_sweep.py
"""

import json
import os
import time

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from mismatch_sweep import (
    make_builds,
    generate_flights_for_build,
    build_feature_rows,
    evaluate_model,
)
from schema import SENSOR_FIELDS, HEALTHY_RANGES

MODEL_PATH = "models/classification/xgboost_fault_classifier_fixed.json"
LEVELS = (0.0, 0.02, 0.05, 0.10, 0.20)   # fraction of each sensor's healthy span
N_UNITS = 4
N_HEALTHY = 3
N_PER_FAULT = 2
OUT_JSON = "docs/offset_noise_sweep.json"


def span(sensor):
    lo, hi = HEALTHY_RANGES[sensor]
    return hi - lo


def apply_offsets(flights, level, seed):
    """Fixed random calibration offset per sensor, shared by all flights of one build."""
    rng = np.random.default_rng(seed)
    offsets = {s: rng.normal(0.0, level * span(s)) for s in SENSOR_FIELDS}
    out = []
    for df in flights:
        d = df.copy()
        for s in SENSOR_FIELDS:
            d[s] = d[s] + offsets[s]
        out.append(d)
    return out


def apply_noise(flights, level, seed):
    """Extra zero-mean measurement noise on every sensor."""
    rng = np.random.default_rng(seed)
    out = []
    for df in flights:
        d = df.copy()
        for s in SENSOR_FIELDS:
            d[s] = d[s] + rng.normal(0.0, level * span(s), size=len(d))
        out.append(d)
    return out


def run(mode, perturb, model, t0):
    rows = []
    for level in LEVELS:
        for build in make_builds(0.0, N_UNITS):     # nominal couplings; only the sensor perturbation varies
            flights = generate_flights_for_build(build, N_HEALTHY, N_PER_FAULT)
            flights = perturb(flights, level, seed=build.seed * 7 + 13)
            feats = build_feature_rows(flights, build.build_id)
            m = evaluate_model(model, feats)
            rows.append({"mode": mode, "level": level, "build_id": build.build_id, **m})
            print(f"[{mode}] level={level:.0%} build={build.build_id} "
                  f"macro_f1={m['macro_f1']:.3f} false_alarm={m['false_alarm_rate']:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    summary = (
        df.groupby("level")
        .agg(macro_f1_mean=("macro_f1", "mean"),
             macro_f1_std=("macro_f1", "std"),
             false_alarm_rate_mean=("false_alarm_rate", "mean"))
        .reset_index()
    )
    print(f"\n=== {mode} ===")
    print(summary.to_string(index=False))
    print()
    return rows


def main():
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    t0 = time.time()

    all_rows = []
    all_rows += run("calibration_offset", apply_offsets, model, t0)
    all_rows += run("extra_noise", apply_noise, model, t0)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"Saved {OUT_JSON}")


if __name__ == "__main__":
    main()