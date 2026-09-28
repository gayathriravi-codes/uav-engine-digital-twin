"""
mismatch_sweep.py -- FULLY WIRED to your real physics_residual.py,
feature_extraction.py, and the trained xgboost_fault_classifier.json.

TWO IMPORTANT FINDINGS FROM YOUR REAL CODE, BEFORE YOU RUN THIS
-----------------------------------------------------------------
1. Your shipped classifier's features are built ENTIRELY from raw sensor
   window stats + a formula-based health_score (feature_extraction.py).
   physics_residual.py exists but its output is never fed into the
   classifier's features today. So this sweep, as-is, tests: "how much
   does classifier accuracy degrade under engine-build mismatch, using
   the pipeline you actually ship." It does NOT (yet) let you make a
   Kavish-0-style "residual representation degrades faster than raw"
   claim, because you only have one representation in production.
   See `train_residual_augmented_variant()` below for an optional,
   clearly-separate experiment that adds physics-residual window stats
   as NEW features and trains a second model to enable that comparison
   for real, without touching your shipped model.

2. train_xgboost.py (as given to me) groups StratifiedGroupKFold on
   `flight_id`, not `base_id`. Per your own project history, `flight_id`
   grouping is the LEAKY split (18 base trajectories relabeled under up
   to 7 fault names each) -- if this is the script that produced your
   shipped `xgboost_fault_classifier.json`, that model's own reported
   CV accuracy is inflated and shouldn't be quoted as-is. This sweep
   doesn't fix that -- it evaluates whatever model file you point it at
   -- but you should confirm which script actually produced your current
   .json before trusting any accuracy number, from this sweep or otherwise.

WHAT ACTUALLY RUNS TODAY
--------------------------
- Data generation at each build-mismatch level: REAL, tested, works.
- Feature extraction + windowing: REAL, using your actual
  extract_window_features / get_window_label / process_flight_df logic.
- Classifier evaluation: REAL, loads your actual xgboost_fault_classifier.json
  and predicts on generated windows.
- The only thing YOU must confirm/fix before trusting numbers: WINDOW_SIZE
  and STRIDE in schema.py (guessed at 30/10 here to match your known
  "30-row sliding window" fact -- confirm against your real schema.py),
  and the class-name-to-int mapping (see CLASS_NAMES below -- must match
  exactly what train_xgboost.py's `sorted(y.unique())` produced).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulator"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "classification"))

from simulator.simulate_healthy import simulate_healthy_flight, NOMINAL_COUPLINGS
from simulator.fault_injectors import INJECTORS
from models.physics_residual import load_baselines, compute_residuals, BASELINE_SPECS
from models.classification.feature_extraction import process_flight_df

# ADAPT: confirm this matches sorted(y.unique()) from your actual training
# data -- it must, since the loaded model's class indices depend on this
# exact order. If your dataset has a slightly different label set, update.
CLASS_NAMES = sorted([
    "cooling_degradation", "misfire", "none", "oil_issue",
    "overheat", "sensor_drift", "vibration_fault",
])
LABEL_TO_INT = {label: i for i, label in enumerate(CLASS_NAMES)}
DROP_COLUMNS = ["label", "flight_id", "window_start", "window_end", "build_id"]


# ---------------------------------------------------------------------------
# 1. Build spec + data generation (same as before, unchanged / verified)
# ---------------------------------------------------------------------------

@dataclass
class EngineBuild:
    build_id: int
    coupling_overrides: Dict[str, float]
    seed: int


def make_builds(spread_pct: float, n_units: int, base_seed: int = 42) -> List[EngineBuild]:
    rng = np.random.default_rng(base_seed)
    builds = []
    for i in range(n_units):
        overrides = {
            field: float(nominal * (1.0 + rng.uniform(-spread_pct, spread_pct)))
            for field, nominal in NOMINAL_COUPLINGS.items()
        }
        builds.append(EngineBuild(build_id=base_seed + i, coupling_overrides=overrides, seed=base_seed + i))
    return builds


def generate_flights_for_build(build: EngineBuild, n_healthy: int, n_per_fault: int,
                                duration: int = 300) -> List[pd.DataFrame]:
    """Returns a LIST of per-flight DataFrames (not concatenated) -- windowing
    must respect individual flight boundaries."""
    flights = []

    for i in range(n_healthy):
        healthy = simulate_healthy_flight(duration_timesteps=duration, seed=build.seed * 1000 + i,
                                           coupling_overrides=build.coupling_overrides)
        healthy["fault_type"] = "none"
        healthy["true_rul_timesteps"] = -1.0
        healthy["fault_onset_idx"] = -1
        healthy["failure_idx"] = -1
        healthy["flight_id"] = f"build{build.build_id}_healthy_{i:03d}"
        flights.append(healthy)

    for fault_type, injector in INJECTORS.items():
        for i in range(n_per_fault):
            healthy = simulate_healthy_flight(duration_timesteps=duration, seed=build.seed * 2000 + i,
                                               coupling_overrides=build.coupling_overrides)
            faulted = injector(healthy, seed=build.seed * 3000 + i)
            faulted["flight_id"] = f"build{build.build_id}_{fault_type}_{i:03d}"
            flights.append(faulted)

    return flights


# ---------------------------------------------------------------------------
# 2. Evaluate the SHIPPED classifier against a build's flights
# ---------------------------------------------------------------------------

def build_feature_rows(flights: List[pd.DataFrame], build_id: int,
                        residual_models=None) -> pd.DataFrame:
    """
    residual_models: if given, adds physics-residual window features
    ("<sensor>_residual_abs_mean") on top of the standard raw features --
    used only by the optional residual-augmented variant, never by the
    default shipped-model evaluation.
    """
    all_rows = []
    for flight_df in flights:
        df = flight_df
        if residual_models is not None:
            df = compute_residuals(df, residual_models)

        rows = process_flight_df(df)

        if residual_models is not None:
            # add simple window-level residual stats, aligned to the same
            # window_start/window_end already computed
            resid_cols = [f"{t}_residual_abs" for t in BASELINE_SPECS if f"{t}_residual_abs" in df.columns]
            for idx, r in rows.iterrows():
                w = df.iloc[int(r["window_start"]):int(r["window_end"]) + 1]
                for c in resid_cols:
                    rows.loc[idx, f"{c}_mean"] = w[c].mean()
                    rows.loc[idx, f"{c}_max"] = w[c].max()

        rows["build_id"] = build_id
        all_rows.append(rows)
    return pd.concat(all_rows, ignore_index=True)


def evaluate_model(model: XGBClassifier, feature_rows: pd.DataFrame) -> Dict[str, float]:
    y_true = feature_rows["label"].map(LABEL_TO_INT)
    if y_true.isna().any():
        unknown = feature_rows.loc[y_true.isna(), "label"].unique()
        raise ValueError(f"Labels not in CLASS_NAMES: {unknown} -- fix CLASS_NAMES at top of file.")

    X = feature_rows.drop(columns=[c for c in DROP_COLUMNS if c in feature_rows.columns])
    preds = model.predict(X)

    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)

    none_idx = LABEL_TO_INT["none"]
    is_true_none = (y_true == none_idx)
    if is_true_none.sum() > 0:
        false_alarm_rate = float((preds[is_true_none] != none_idx).mean())
    else:
        false_alarm_rate = None

    return {"macro_f1": float(macro_f1), "false_alarm_rate": false_alarm_rate}


# ---------------------------------------------------------------------------
# 3. The sweep, against your SHIPPED model
# ---------------------------------------------------------------------------

def run_mismatch_sweep(
    model_path: str = "models/classification/xgboost_fault_classifier_aug.json",
    spread_levels_pct=(0.0, 0.02, 0.05, 0.08, 0.10, 0.15),
    n_units_per_level: int = 6,
    n_healthy_per_unit: int = 5,
    n_per_fault_per_unit: int = 3,
    min_macro_f1_at_5pct: Optional[float] = None,
    out_json: str = "docs/mismatch_sweep_results.json",
) -> pd.DataFrame:
    model = XGBClassifier()
    model.load_model(model_path)

    rows = []
    for spread in spread_levels_pct:
        builds = make_builds(spread, n_units_per_level)
        for build in builds:
            flights = generate_flights_for_build(build, n_healthy_per_unit, n_per_fault_per_unit)
            feature_rows = build_feature_rows(flights, build.build_id)
            metrics = evaluate_model(model, feature_rows)
            rows.append({"spread_pct": spread, "build_id": build.build_id, **metrics})

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("spread_pct")
        .agg(macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
             false_alarm_rate_mean=("false_alarm_rate", "mean"))
        .reset_index()
    )

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(json.loads(summary.to_json(orient="records")), f, indent=2)

    if min_macro_f1_at_5pct is not None:
        at5 = summary[summary.spread_pct == 0.05]
        if not at5.empty and at5.iloc[0]["macro_f1_mean"] < min_macro_f1_at_5pct:
            print(f"[MISMATCH SWEEP] Below floor at 5% spread: "
                  f"{at5.iloc[0]['macro_f1_mean']:.3f} < {min_macro_f1_at_5pct}. "
                  f"Report this honestly.")

    print(summary.to_string(index=False))
    return summary


# ---------------------------------------------------------------------------
# 4. OPTIONAL: residual-augmented variant, to enable the real Kavish-0
#    comparison. Trains a SEPARATE model, never touches your shipped one.
# ---------------------------------------------------------------------------

def train_residual_augmented_variant(nominal_flights: List[pd.DataFrame]):
    """
    ADAPT: this is a sketch, not run/tested here (needs your real
    physics_baselines.joblib, fit on your real nominal-build training
    data). Fits a SECOND xgboost model on features = raw window stats +
    physics-residual window stats, trained only on nominal-build (spread=0)
    data -- exactly mirroring how your shipped model was trained, just with
    extra features. Then run_mismatch_sweep-style evaluation can be repeated
    for THIS model, and the two macro-F1-vs-spread curves compared directly.
    This is the only way to make the "residual representation more/less
    robust than raw" claim honestly, since it doesn't exist as a real
    comparison in your shipped pipeline today.
    """
    raise NotImplementedError("Sketch only -- wire to your real training loop if you want this comparison.")


if __name__ == "__main__":
    print(__doc__)
