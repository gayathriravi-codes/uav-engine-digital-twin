"""
anomaly_detector.py
--------------------
Unsupervised multivariate anomaly detector on physics-residual features.

Why this exists: sensor_trust.py catches a sensor lying about itself
(impossible value, flatline, jump, mean-shift). It cannot catch a
*combination* of sensors that's individually plausible but jointly
wrong (e.g. EGT and fuel_flow both sit inside PHYSICAL_LIMITS but their
ratio is physically inconsistent). An IsolationForest trained on
residuals - not raw sensor values - is scored against the trajectory-
grouped train/test split, exactly like the RUL and fault models, so it
inherits the same leakage guarantees rather than reintroducing the bug
we already found and fixed once.

This is explicitly NOT a replacement for sensor_trust.py or the XGBoost
fault classifier. It runs alongside them and flags "something is
jointly off here" for cases neither of the other two catches.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

from schema import SENSOR_FIELDS, HEALTHY_RANGES

# Residual features: (value - healthy_midpoint) / healthy_halfwidth,
# so every sensor is on a comparable unitless scale before the
# IsolationForest sees it. This also makes the model's behavior
# explainable: a residual of 2.0 means "2x the healthy band's half-width
# away from the healthy midpoint", regardless of sensor units.
def compute_residual_features(df: pd.DataFrame) -> pd.DataFrame:
    residuals = {}
    for sensor in SENSOR_FIELDS:
        lo, hi = HEALTHY_RANGES[sensor]
        mid = (lo + hi) / 2.0
        halfwidth = (hi - lo) / 2.0
        residuals[f"{sensor}_residual"] = (df[sensor] - mid) / halfwidth
    return pd.DataFrame(residuals, index=df.index)


class ResidualAnomalyDetector:
    def __init__(self, n_estimators=200, contamination="auto", random_state=42):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        self.feature_cols = [f"{s}_residual" for s in SENSOR_FIELDS]
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame):
        """
        train_df must contain ONLY rows from the training split of your
        trajectory-grouped split (fix_trajectory_split.py) - fitting this
        on the full dataset before splitting would leak test trajectories
        into the anomaly baseline, the same bug class as the original
        108-flights leak.
        """
        feats = compute_residual_features(train_df)
        X = self.scaler.fit_transform(feats[self.feature_cols])
        self.model.fit(X)
        self.is_fitted = True
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns anomaly_score (higher = more anomalous, unlike sklearn's
        raw convention) and is_anomaly (bool) per row."""
        if not self.is_fitted:
            raise RuntimeError("Call .fit() on the TRAIN split before scoring.")
        feats = compute_residual_features(df)
        X = self.scaler.transform(feats[self.feature_cols])
        raw_score = self.model.score_samples(X)       # higher = more normal
        anomaly_score = -raw_score                     # flip: higher = more anomalous
        is_anomaly = self.model.predict(X) == -1
        out = df.copy()
        out["anomaly_score"] = anomaly_score
        out["is_anomaly"] = is_anomaly
        return out

    def save(self, path="models/anomaly_detector.joblib"):
        joblib.dump({"scaler": self.scaler, "model": self.model,
                     "feature_cols": self.feature_cols}, path)

    @classmethod
    def load(cls, path="models/anomaly_detector.joblib"):
        data = joblib.load(path)
        obj = cls()
        obj.scaler = data["scaler"]
        obj.model = data["model"]
        obj.feature_cols = data["feature_cols"]
        obj.is_fitted = True
        return obj


if __name__ == "__main__":
    # Example usage - adjust the split loading to match fix_trajectory_split.py's
    # actual output once you paste it; this assumes it produces train_df/test_df.
    import glob
    all_files = glob.glob("data/raw/*.csv")
    df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)

    # PLACEHOLDER split - replace with your real trajectory-grouped split
    healthy_only = df[df["fault_type"] == "none"]
    detector = ResidualAnomalyDetector().fit(healthy_only)
    scored = detector.score(df)
    print(scored.groupby("fault_type")["anomaly_score"].describe())