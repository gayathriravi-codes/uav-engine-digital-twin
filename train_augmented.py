"""
train_augmented.py

Retrains the fault classifier with sensor noise / calibration-offset
augmentation, using the same base_id-grouped CV as train_xgboost_fixed.py.

- Augmentation is applied to TRAINING groups only.
- Each augmented copy keeps its source flight's base_id (no leakage).
- Held-out folds are scored on clean data and on two perturbed variants
  (noise 5%, offset 5% of each sensor's healthy span) that are never used
  in training.
- Saves models/classification/xgboost_fault_classifier_aug.json

Run from the project root:
    python -u train_augmented.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from schema import SENSOR_FIELDS, HEALTHY_RANGES
from models.classification.feature_extraction import process_flight_df

RAW_DIR = PROJECT_ROOT / "data" / "raw"
FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "classification_features.csv"
MODEL_OUT = PROJECT_ROOT / "models" / "classification" / "xgboost_fault_classifier_aug.json"

N_AUG = 3            # augmented copies per flight
NOISE_MAX = 0.08     # augmentation range, fraction of healthy span
OFFSET_MAX = 0.10
EVAL_NOISE = 0.05    # held-out evaluation perturbations (never trained on)
EVAL_OFFSET = 0.05
SEED = 42

DROP_COLUMNS = ["label", "flight_id", "window_start", "window_end", "base_id"]


def span(sensor):
    lo, hi = HEALTHY_RANGES[sensor]
    return hi - lo


def perturb(df, noise_level, offset_level, rng):
    d = df.copy()
    for s in SENSOR_FIELDS:
        if offset_level > 0:
            d[s] = d[s] + rng.normal(0.0, offset_level * span(s))
        if noise_level > 0:
            d[s] = d[s] + rng.normal(0.0, noise_level * span(s), size=len(d))
    return d


def compute_base_id(flight_id):
    suffix = flight_id.rsplit("_", 1)[-1]
    return suffix if suffix.isdigit() else flight_id


def window_variant(df, flight_id, variant):
    rows = process_flight_df(df)
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return None
    out["flight_id"] = flight_id
    out["base_id"] = compute_base_id(flight_id)
    out["variant"] = variant
    return out


def build_table():
    rng = np.random.default_rng(SEED)
    frames = []
    files = sorted(RAW_DIR.glob("*.csv"))
    t0 = time.time()
    for k, path in enumerate(files, start=1):
        df = pd.read_csv(path)
        fid = path.stem

        f = window_variant(df, fid, "clean")
        if f is not None:
            frames.append(f)

        for a in range(N_AUG):
            mode = rng.choice(["noise", "offset", "both"])
            noise = rng.uniform(0, NOISE_MAX) if mode in ("noise", "both") else 0.0
            offset = rng.uniform(0, OFFSET_MAX) if mode in ("offset", "both") else 0.0
            f = window_variant(perturb(df, noise, offset, rng), fid, "aug")
            if f is not None:
                frames.append(f)

        f = window_variant(perturb(df, EVAL_NOISE, 0.0, rng), fid, "eval_noise")
        if f is not None:
            frames.append(f)
        f = window_variant(perturb(df, 0.0, EVAL_OFFSET, rng), fid, "eval_offset")
        if f is not None:
            frames.append(f)

        if k % 10 == 0 or k == len(files):
            print(f"  windowed {k}/{len(files)} flights ({time.time() - t0:.0f}s)", flush=True)
    return pd.concat(frames, ignore_index=True)


def make_model(n_classes):
    return XGBClassifier(
        objective="multi:softprob", num_class=n_classes,
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        eval_metric="mlogloss", random_state=42, n_jobs=-1,
    )


def main():
    ref_cols = [c for c in pd.read_csv(FEATURES_CSV, nrows=1).columns if c not in DROP_COLUMNS]
    print(f"Reference feature columns: {len(ref_cols)}")

    print("Building augmented window table from raw flights...")
    table = build_table()
    missing = [c for c in ref_cols if c not in table.columns]
    if missing:
        raise SystemExit(f"Feature columns missing from process_flight_df output: {missing[:5]} ...")
    print(f"Table: {len(table)} windows; variants: {table['variant'].value_counts().to_dict()}")

    class_names = sorted(table.loc[table["variant"] == "clean", "label"].unique())
    label_to_int = {c: i for i, c in enumerate(class_names)}
    print("Classes:", class_names)
    if len(class_names) != 7:
        print("WARNING: expected 7 classes -- check before trusting anything below.")

    clean = table[table["variant"] == "clean"].reset_index(drop=True)
    y_clean = clean["label"].map(label_to_int)
    groups_clean = clean["base_id"]

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    all_true, all_pred = [], []

    for fold, (tr_idx, te_idx) in enumerate(cv.split(clean, y_clean, groups=groups_clean), start=1):
        train_groups = set(groups_clean.iloc[tr_idx])
        test_groups = set(groups_clean.iloc[te_idx])
        assert not (train_groups & test_groups), "LEAKAGE: base_id in both train and test"

        train = table[table["base_id"].isin(train_groups) & table["variant"].isin(["clean", "aug"])]
        model = make_model(len(class_names))
        model.fit(train[ref_cols], train["label"].map(label_to_int))

        row = {"fold": fold}
        for variant in ["clean", "eval_noise", "eval_offset"]:
            test = table[table["base_id"].isin(test_groups) & (table["variant"] == variant)]
            pred = model.predict(test[ref_cols])
            true = test["label"].map(label_to_int)
            row[f"{variant}_acc"] = accuracy_score(true, pred)
            row[f"{variant}_f1"] = f1_score(true, pred, average="macro", zero_division=0)
            if variant == "clean":
                all_true.extend(true)
                all_pred.extend(pred)
        results.append(row)
        print(f"Fold {fold}: clean F1={row['clean_f1']:.3f}  "
              f"noise5% F1={row['eval_noise_f1']:.3f}  offset5% F1={row['eval_offset_f1']:.3f}", flush=True)

    res = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("5-FOLD CV (grouped by base_id), trained WITH augmentation")
    print("=" * 60)
    for variant, label in [("clean", "clean held-out"),
                           ("eval_noise", "held-out + 5% noise"),
                           ("eval_offset", "held-out + 5% offset")]:
        print(f"{label:24s} acc {res[variant + '_acc'].mean():.4f} +/- {res[variant + '_acc'].std():.4f}"
              f"   macro F1 {res[variant + '_f1'].mean():.4f} +/- {res[variant + '_f1'].std():.4f}")

    print("\nPer-class report on CLEAN held-out windows:")
    print(classification_report(all_true, all_pred, target_names=class_names, zero_division=0))

    print("Training final model on all clean + augmented windows...")
    final_train = table[table["variant"].isin(["clean", "aug"])]
    final = make_model(len(class_names))
    final.fit(final_train[ref_cols], final_train["label"].map(label_to_int))
    final.save_model(str(MODEL_OUT))
    print(f"Saved {MODEL_OUT}")


if __name__ == "__main__":
    main()