"""
stress_sweep.py -- same as run_mismatch_sweep but prints progress per build
and writes results incrementally, so you can see it working (and keep partial
results if you stop it).

Run from the project root:
    python -u stress_sweep.py
"""

import json
import os
import time

import pandas as pd
from xgboost import XGBClassifier

from mismatch_sweep import (
    make_builds,
    generate_flights_for_build,
    build_feature_rows,
    evaluate_model,
)

MODEL_PATH = "models/classification/xgboost_fault_classifier_fixed.json"
SPREADS = (0.0, 0.30, 0.40)      # add 0.15 / 0.20 later if you want
N_UNITS = 3
N_HEALTHY = 3
N_PER_FAULT = 2
OUT_JSON = "docs/mismatch_sweep_stress.json"


def main():
    model = XGBClassifier()
    model.load_model(MODEL_PATH)

    rows = []
    t0 = time.time()
    total = len(SPREADS) * N_UNITS
    done = 0

    for spread in SPREADS:
        for build in make_builds(spread, N_UNITS):
            flights = generate_flights_for_build(build, N_HEALTHY, N_PER_FAULT)
            feats = build_feature_rows(flights, build.build_id)
            m = evaluate_model(model, feats)
            rows.append({"spread_pct": spread, "build_id": build.build_id, **m})
            done += 1
            print(f"[{done}/{total}] spread={spread:.0%} build={build.build_id} "
                  f"macro_f1={m['macro_f1']:.3f} "
                  f"false_alarm={m['false_alarm_rate']:.3f} "
                  f"({time.time() - t0:.0f}s elapsed)", flush=True)

            # save partial results after every build
            os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
            with open(OUT_JSON, "w") as f:
                json.dump(rows, f, indent=2)

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("spread_pct")
        .agg(macro_f1_mean=("macro_f1", "mean"),
             macro_f1_std=("macro_f1", "std"),
             false_alarm_rate_mean=("false_alarm_rate", "mean"))
        .reset_index()
    )
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()