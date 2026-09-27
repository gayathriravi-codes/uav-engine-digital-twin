"""
anomaly_detector_eval.py
---------------------------
Proper trajectory-grouped refit and evaluation of the physics-residual
anomaly detector, replacing the earlier placeholder version that fit on
ALL "none"-labeled rows regardless of trajectory -- the same leakage
shape as the bug found and fixed in the RUL and classification
pipelines this session, just not yet applied here.

Split: base trajectories (000-017, via the same get_base_id convention
used throughout this session) are split into train/test groups BEFORE
anything is fit. The detector is fit ONLY on "none"-labeled rows from
TRAIN trajectories. It is then scored on ALL rows (every fault type +
none) from TEST trajectories only -- trajectories the detector never
saw in any form during fitting.

Evaluation: rather than reporting mean anomaly scores per fault type
(which hides how many real false positives/negatives a chosen threshold
would produce), this reports precision/recall at three threshold
choices, chosen as percentiles of the TRAIN "none" distribution itself
(90th/95th/99th) -- so thresholds are set the way they would be in
practice (calibrated on data known to be healthy), not tuned against
the test set's answers.

Run: python anomaly_detector_eval.py   (from project root)
"""
import sys
import os
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from schema import SENSOR_FIELDS, HEALTHY_RANGES

RAW_DIR = "data/raw"


def get_base_id(flight_id: str) -> str:
    """Same convention as test_leakage.py / train_xgboost_fixed.py."""
    return flight_id.rsplit("_", 1)[-1]


def compute_residual_features(df: pd.DataFrame) -> pd.DataFrame:
    residuals = {}
    for sensor in SENSOR_FIELDS:
        lo, hi = HEALTHY_RANGES[sensor]
        mid = (lo + hi) / 2.0
        halfwidth = (hi - lo) / 2.0
        residuals[f"{sensor}_residual"] = (df[sensor] - mid) / halfwidth
    return pd.DataFrame(residuals, index=df.index)


def load_all_raw_flights():
    """
    Loads every flight CSV in data/raw/ that has a real flight_id and
    fault_type (i.e. the original 108+123-style dataset), EXCLUDING the
    new mission_*.csv scenario files -- those don't share the same
    base-trajectory structure (18 trajectories x N fault labels) that
    this grouped split depends on, and mixing them in would silently
    break the leakage-safety guarantee this script exists to provide.
    """
    frames = []
    for path in sorted(glob.glob(os.path.join(RAW_DIR, "*.csv"))):
        name = os.path.basename(path)
        if name.startswith("mission_"):
            continue
        df = pd.read_csv(path)
        if "flight_id" not in df.columns or "fault_type" not in df.columns:
            continue
        frames.append(df)
    if not frames:
        raise RuntimeError(
            f"No usable flight CSVs found in {RAW_DIR}/ -- expected the original "
            f"108+123-flight dataset (flight_id + fault_type columns required)."
        )
    return pd.concat(frames, ignore_index=True)


def grouped_train_test_split(df, test_frac=0.3, seed=42):
    """
    Splits by BASE trajectory, not by flight_id or by row -- the same
    principle as every other split fixed this session. Returns
    (train_df, test_df).
    """
    base_ids = df["flight_id"].apply(get_base_id)
    unique_bases = sorted(base_ids.unique())

    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(unique_bases)
    n_test = max(1, int(len(unique_bases) * test_frac))
    test_bases = set(shuffled[:n_test])
    train_bases = set(shuffled[n_test:])

    overlap = train_bases & test_bases
    assert not overlap, f"Split bug: {overlap} in both train and test"

    train_df = df[base_ids.isin(train_bases)].copy()
    test_df = df[base_ids.isin(test_bases)].copy()
    return train_df, test_df, sorted(train_bases), sorted(test_bases)


def evaluate_at_threshold(test_scored, threshold, fault_types):
    """
    Computes precision/recall treating "any fault_type != none" as the
    positive class, PLUS per-fault-type recall individually (which
    fault types the detector actually catches, not just an aggregate).
    """
    is_anomaly = test_scored["anomaly_score"] >= threshold
    is_actual_fault = test_scored["fault_type"] != "none"

    tp = (is_anomaly & is_actual_fault).sum()
    fp = (is_anomaly & ~is_actual_fault).sum()
    fn = (~is_anomaly & is_actual_fault).sum()
    tn = (~is_anomaly & ~is_actual_fault).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else float("nan")

    per_fault_recall = {}
    for ft in fault_types:
        if ft == "none":
            continue
        mask = test_scored["fault_type"] == ft
        n = mask.sum()
        if n == 0:
            continue
        caught = (is_anomaly & mask).sum()
        per_fault_recall[ft] = (caught / n, n)

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": false_positive_rate,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "per_fault_recall": per_fault_recall,
    }


if __name__ == "__main__":
    print("Loading raw flight data (excluding mission_*.csv scenario files)...")
    df = load_all_raw_flights()
    print(f"Total rows: {len(df)}  |  fault_type counts:\n{df['fault_type'].value_counts()}\n")

    print("Splitting by BASE trajectory (leakage-safe, same convention as test_leakage.py)...")
    train_df, test_df, train_bases, test_bases = grouped_train_test_split(df, test_frac=0.3, seed=42)
    print(f"Train base trajectories ({len(train_bases)}): {train_bases}")
    print(f"Test base trajectories  ({len(test_bases)}): {test_bases}\n")

    # Fit ONLY on healthy rows from TRAIN trajectories.
    train_healthy = train_df[train_df["fault_type"] == "none"]
    print(f"Fitting IsolationForest on {len(train_healthy)} healthy rows from TRAIN trajectories only...")

    feature_cols = [f"{s}_residual" for s in SENSOR_FIELDS]
    train_feats = compute_residual_features(train_healthy)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_feats[feature_cols])

    model = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
    model.fit(X_train)

    # Score ALL of test (every fault type + none), from trajectories
    # never seen during fitting in any form.
    print(f"Scoring {len(test_df)} rows from {len(test_bases)} held-out TEST trajectories "
          f"(never seen during fitting)...\n")
    test_feats = compute_residual_features(test_df)
    X_test = scaler.transform(test_feats[feature_cols])
    raw_score = model.score_samples(X_test)
    test_scored = test_df.copy()
    test_scored["anomaly_score"] = -raw_score  # flip: higher = more anomalous

    # Thresholds set from the TRAIN healthy distribution's own percentiles
    # -- i.e. calibrated the way you'd actually do it in practice (you
    # don't get to peek at test-set fault labels to pick a threshold).
    train_healthy_scores = -model.score_samples(X_train)
    thresholds = {
        "90th percentile of train-healthy scores": np.percentile(train_healthy_scores, 90),
        "95th percentile of train-healthy scores": np.percentile(train_healthy_scores, 95),
        "99th percentile of train-healthy scores": np.percentile(train_healthy_scores, 99),
    }

    fault_types = sorted(df["fault_type"].unique())

    print("=" * 70)
    print("EVALUATION ON HELD-OUT TEST TRAJECTORIES")
    print("(thresholds calibrated on TRAIN healthy data only, never on test labels)")
    print("=" * 70)

    for label, thresh in thresholds.items():
        result = evaluate_at_threshold(test_scored, thresh, fault_types)
        print(f"\n--- Threshold: {label} (score >= {thresh:.4f}) ---")
        print(f"  Overall: precision={result['precision']:.3f}  recall={result['recall']:.3f}  "
              f"false_positive_rate={result['false_positive_rate']:.3f}")
        print(f"  Confusion: TP={result['tp']}  FP={result['fp']}  FN={result['fn']}  TN={result['tn']}")
        print(f"  Per-fault-type recall (fraction of that fault's rows flagged as anomalous):")
        for ft, (recall_val, n) in sorted(result["per_fault_recall"].items()):
            print(f"    {ft:22s} recall={recall_val:.3f}  (n={n})")

    print("\n" + "=" * 70)
    print("HONEST SUMMARY")
    print("=" * 70)
    print("This detector is UNSUPERVISED (never trained on fault labels) and is")
    print("evaluated on trajectories it never saw, healthy or faulty, in any form.")
    print("It is meant to complement the supervised XGBoost classifier (91.1% accuracy,")
    print("leak-fixed) by catching deviations the classifier wasn't trained to name --")
    print("not to replace it. Compare the per-fault recall above against the")
    print("classifier's per-class recall from earlier (train_xgboost_fixed.py's")
    print("classification report) to see where each model's strengths differ.")