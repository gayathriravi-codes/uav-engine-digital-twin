"""
physics_residual.py  (v2 - matched to real AeroTwin schema)
-------------------------------------------------------------
Real columns confirmed from data/raw/*.csv:
  timestamp, rpm, egt, cht, oil_pressure, oil_temp, vibration,
  fuel_flow, fault_type, true_rul_timesteps, fault_onset_idx,
  failure_idx, flight_id

Healthy rows = fault_type == 'none'. This includes ALL rows in
healthy_*.csv, AND all pre-onset rows in fault files (fault_onset_idx
marks where the fault begins - anything before it is still healthy
engine behavior even in a "faulty" flight file).

Baseline pairs chosen from what actually exists in the data (no
throttle/ambient_temperature - those don't exist in this simulator):
  - egt   expected from [rpm, fuel_flow]   (combustion temp driven by
          burn rate + engine speed)
  - cht   expected from [rpm, oil_temp]    (cylinder head temp tracks
          engine speed + overall thermal state)
  - oil_pressure expected from [rpm]       (oil pressure is largely
          RPM-driven in a healthy engine - also matches your existing
          RUL_CORE_SENSORS finding that oil_pressure matters)
"""

import glob
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

BASELINE_SPECS = {
    "egt": ["rpm", "fuel_flow"],
    "cht": ["rpm", "oil_temp"],
    "oil_pressure": ["rpm"],
}

RAW_DATA_GLOB = "data/raw/*.csv"


def load_all_healthy_rows(raw_glob=RAW_DATA_GLOB):
    """
    Loads every raw CSV and keeps only rows where fault_type == 'none'.
    This pulls healthy rows from healthy_*.csv AND from the pre-onset
    portion of every fault file, giving a much larger healthy training
    set than the 15 healthy files alone.
    """
    files = glob.glob(raw_glob)
    if not files:
        raise FileNotFoundError(f"No CSVs found matching {raw_glob} - "
                                 f"run this from the AeroTwin project root.")

    healthy_frames = []
    for f in files:
        df = pd.read_csv(f)
        healthy_frames.append(df[df["fault_type"] == "none"])

    healthy_df = pd.concat(healthy_frames, ignore_index=True)
    print(f"[physics_residual] loaded {len(files)} files, "
          f"{len(healthy_df)} healthy rows total")
    return healthy_df


def fit_healthy_baselines(healthy_df, specs=BASELINE_SPECS):
    models = {}
    for target, predictors in specs.items():
        X = healthy_df[predictors].values
        y = healthy_df[target].values
        model = LinearRegression().fit(X, y)
        models[target] = model
        r2 = model.score(X, y)
        print(f"[physics_residual] fitted baseline for {target} "
              f"using {predictors} -> R^2={r2:.3f}")
        if r2 < 0.3:
            print(f"  WARNING: R^2 is weak for {target} - consider "
                  f"dropping this baseline from the deck claim, or "
                  f"trying different/additional predictors.")
    return models


def save_baselines(models, path="models/physics_baselines.joblib"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(models, path)
    print(f"[physics_residual] saved baselines -> {path}")


def load_baselines(path="models/physics_baselines.joblib"):
    return joblib.load(path)


def compute_residuals(df, models, specs=BASELINE_SPECS):
    df = df.copy()
    for target, predictors in specs.items():
        if target not in models:
            continue
        X = df[predictors].values
        expected = models[target].predict(X)
        residual = df[target].values - expected
        df[f"{target}_residual"] = residual
        df[f"{target}_residual_abs"] = np.abs(residual)
    return df


def physics_anomaly_score(df, models, specs=BASELINE_SPECS, healthy_stats=None):
    """
    healthy_stats: optional dict {sensor: std_of_residual_on_healthy_data}
    from compute_residual_healthy_stats() below - use this instead of the
    batch min-max normalization for a stable, deployment-ready score
    (a batch-relative score changes meaning depending on what's in the
    batch, which you don't want for a dashboard).
    """
    df = compute_residuals(df, models, specs)
    residual_cols = [f"{t}_residual_abs" for t in specs if f"{t}_residual_abs" in df.columns]
    if not residual_cols:
        return df

    if healthy_stats:
        z_cols = []
        for t in specs:
            col = f"{t}_residual_abs"
            if col in df.columns and t in healthy_stats:
                z = df[col] / (healthy_stats[t] + 1e-9)
                df[f"{t}_residual_z"] = z
                z_cols.append(f"{t}_residual_z")
        score = 100 - (df[z_cols].mean(axis=1) / 4.0 * 100)
    else:
        norm = df[residual_cols].apply(lambda c: (c - c.min()) / (c.max() - c.min() + 1e-9))
        score = 100 - (norm.mean(axis=1) * 100)

    df["physics_health_score"] = score.clip(0, 100)
    return df


def compute_residual_healthy_stats(healthy_df, models, specs=BASELINE_SPECS):
    """Run once after fitting baselines - gives stable healthy_stats for
    physics_anomaly_score() so scores don't shift meaning batch-to-batch."""
    scored = compute_residuals(healthy_df, models, specs)
    stats = {}
    for t in specs:
        col = f"{t}_residual_abs"
        if col in scored.columns:
            stats[t] = scored[col].std()
    print(f"[physics_residual] healthy residual stds: {stats}")
    return stats


if __name__ == "__main__":
    healthy_df = load_all_healthy_rows()
    models = fit_healthy_baselines(healthy_df)
    save_baselines(models)
    stats = compute_residual_healthy_stats(healthy_df, models)
    joblib.dump(stats, "models/physics_healthy_stats.joblib")
    print("[physics_residual] done. Baselines + healthy stats saved to models/")