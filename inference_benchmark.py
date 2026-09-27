"""
inference_benchmark.py
------------------------
Product-readiness benchmark: measures raw inference speed and model
footprint for the fault classifier and RUL ensemble, and folds in the
detection-latency finding from onset_latency_check.py (item 10 work) -
"how fast is a single prediction" and "how fast does the system notice
a real fault" are different claims, and both matter for the deck/
Model Validation Report.
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.classification.predict import predict_fault, load_model, MODEL_PATH
from models.rul.train_rul_ensemble import load_ensemble, build_inference_window, predict_rul_ensemble

N_TIMING_RUNS = 50
TEST_FLIGHT = "data/raw/overheat_000.csv"
WINDOW_SIZE = 30

REQUIRED_SENSORS = [
    "rpm", "egt", "cht", "oil_pressure",
    "oil_temp", "vibration", "fuel_flow",
]


def format_bytes(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return str(round(n, 1)) + " " + unit
        n /= 1024
    return str(round(n, 1)) + " TB"


def benchmark_fault_classifier(df):
    window = df.iloc[-WINDOW_SIZE:].copy()

    predict_fault(window)

    times = []
    for _ in range(N_TIMING_RUNS):
        start = time.perf_counter()
        predict_fault(window)
        times.append(time.perf_counter() - start)

    return {
        "mean_ms": np.mean(times) * 1000,
        "p95_ms": np.percentile(times, 95) * 1000,
        "min_ms": np.min(times) * 1000,
        "max_ms": np.max(times) * 1000,
    }


def benchmark_rul_ensemble(df):
    window = df.iloc[-WINDOW_SIZE:].copy()
    raw_window = window[REQUIRED_SENSORS].to_numpy()
    inference_window = build_inference_window(raw_window)

    models, scaler, dropped_idx_list = load_ensemble()

    predict_rul_ensemble(inference_window, models, scaler, dropped_idx_list, calibrated=True)

    times = []
    for _ in range(N_TIMING_RUNS):
        start = time.perf_counter()
        predict_rul_ensemble(inference_window, models, scaler, dropped_idx_list, calibrated=True)
        times.append(time.perf_counter() - start)

    return {
        "mean_ms": np.mean(times) * 1000,
        "p95_ms": np.percentile(times, 95) * 1000,
        "min_ms": np.min(times) * 1000,
        "max_ms": np.max(times) * 1000,
    }


def get_model_sizes():
    sizes = {}

    if MODEL_PATH.exists():
        sizes["xgboost_fault_classifier"] = MODEL_PATH.stat().st_size

    rul_dir = PROJECT_ROOT / "models" / "rul"
    if rul_dir.exists():
        rul_total = 0
        rul_files = []
        for ext in ("*.pt", "*.pth", "*.joblib", "*.h5", "*.json", "*.pkl"):
            for f in rul_dir.glob(ext):
                if "output" in f.name or f.suffix == ".txt":
                    continue
                rul_total += f.stat().st_size
                rul_files.append(f.name)
        if rul_total > 0:
            label = "rul_ensemble (" + ", ".join(rul_files) + ")"
            sizes[label] = rul_total

    return sizes


def main():
    print("=" * 78)
    print("AEROTWIN INFERENCE FOOTPRINT BENCHMARK")
    print("=" * 78)

    df = pd.read_csv(TEST_FLIGHT)

    print("")
    print("Test flight: " + TEST_FLIGHT)
    print("Timing runs per model: " + str(N_TIMING_RUNS))

    print("\n--- Fault Classifier (XGBoost) ---")
    clf_timing = benchmark_fault_classifier(df)
    for k, v in clf_timing.items():
        print(k.ljust(10) + ": " + format(v, ".2f") + " ms")

    print("\n--- RUL Ensemble ---")
    rul_timing = benchmark_rul_ensemble(df)
    for k, v in rul_timing.items():
        print(k.ljust(10) + ": " + format(v, ".2f") + " ms")

    total_mean_ms = clf_timing["mean_ms"] + rul_timing["mean_ms"]
    print("")
    print("Combined mean per-cycle inference time: " + format(total_mean_ms, ".2f") + " ms")

    print("\n--- Model Sizes ---")
    sizes = get_model_sizes()
    for name, size in sizes.items():
        print(name.ljust(50) + ": " + format_bytes(size))

    print("\n" + "=" * 78)
    print("DETECTION LATENCY (separate from raw inference speed - see onset_latency_check.py)")
    print("=" * 78)
    print("""
Raw single-call inference is fast (see above), but time-to-DETECT a real
fault onset is a different, larger number - it depends on how many ticks
of fault-affected data must accumulate in the classifier's 30-row window
before the aggregate features cross the decision boundary. From 9-flight
validation (onset_latency_check.py):

  Fault type    Raw detection latency post-onset
  ----------    ---------------------------------
  overheat      ~16-22 ticks  (~80-110 timesteps at step=5)
  oil_issue     ~16-22 ticks  (~80-110 timesteps at step=5)
  misfire       ~87-103 ticks (~435-515 timesteps at step=5)

Misfire's latency is structurally higher because it is an intermittent
fault (periodic spikes, not sustained degradation), so the fixed 30-row
aggregate-feature window takes far longer to fill with fault-affected
rows than for a sustained fault like overheat/oil_issue. This is a
known, documented limitation of window-based aggregate classification
against intermittent faults - see Model Validation Report.
""")


if __name__ == "__main__":
    main()