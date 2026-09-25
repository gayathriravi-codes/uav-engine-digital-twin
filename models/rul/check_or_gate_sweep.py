"""
check_or_gate_sweep.py -- test whether adding an ensemble-std condition to
the OOD gate (is_ood if z_distance > 2.0 OR std > threshold) fixes bucket
(0,50) coverage WITHOUT breaking the other buckets, which currently pass
with zero std-based flags.

Sweeps a range of std thresholds on the FULL validation set (all buckets),
for each one reporting:
  - per-bucket coverage (does (0,50) clear 90%? do the others stay >= 90%?)
  - how many windows newly get flagged (global + per bucket)

Run: python check_or_gate_sweep.py
"""
import os
import sys

import numpy as np
import torch
from sklearn.model_selection import train_test_split
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rul import load_all_flights, MODEL_OUT_DIR
from train_rul_ensemble import (
    RULRegressorVariant, build_windowed_dataset_v5, HIDDEN_SIZES, N_VARIANTS,
    compute_ood_zscore_distance, predict_rul_ensemble,
)
from calibrate_rul import load_calibration_params, apply_calibration, _bucket_index

Z_THRESHOLD = 2.0
STD_THRESHOLDS_TO_TRY = [12.0, 15.0, 18.0, 20.0]
BUCKETS = [(0, 50), (50, 100), (100, 200), (200, 350)]


def ensemble_point_and_std(window, models, scaler, dropped_idx_list):
    scaled = scaler.transform(window.reshape(-1, window.shape[-1])).reshape(window.shape)
    preds = []
    with torch.no_grad():
        for model, dropped_idx in zip(models, dropped_idx_list):
            model.eval()
            wv = scaled.copy()
            if len(dropped_idx) > 0:
                wv[:, dropped_idx] = 0.0
            x = torch.tensor(wv, dtype=torch.float32).unsqueeze(0)
            preds.append(model(x).item())
    preds = np.array(preds)
    return max(0.0, float(preds.mean())), float(preds.std())


if __name__ == "__main__":
    flights = load_all_flights()
    X, y, flight_ids = build_windowed_dataset_v5(flights)

    unique_flights = np.unique(flight_ids)
    train_flights, temp_flights = train_test_split(unique_flights, test_size=0.3, random_state=42)
    val_flights, test_flights = train_test_split(temp_flights, test_size=0.5, random_state=42)
    val_mask = np.isin(flight_ids, val_flights)
    X_val, y_val = X[val_mask], y[val_mask]
    y_val = np.asarray(y_val, dtype=float)
    print(f"Validation windows: {len(X_val)} (val set, NOT test -- safe to re-run)\n")

    scaler = joblib.load(os.path.join(MODEL_OUT_DIR, "rul_ensemble_scaler.joblib"))
    dropped_idx_list = joblib.load(os.path.join(MODEL_OUT_DIR, "rul_ensemble_dropped_idx.joblib"))
    models = []
    for i in range(N_VARIANTS):
        model = RULRegressorVariant(hidden_size=HIDDEN_SIZES[i])
        model.load_state_dict(torch.load(os.path.join(MODEL_OUT_DIR, f"rul_model_variant_{i}.pt")))
        model.eval()
        models.append(model)

    params = load_calibration_params()
    bucket_edges = params["bucket_edges"]

    # Precompute once: point estimate, std, z-distance, bucket id for every window.
    print("Precomputing point estimate / std / z-distance for all validation windows ...")
    point_ests = np.zeros(len(X_val))
    stds = np.zeros(len(X_val))
    z_dists = np.zeros(len(X_val))
    for i in range(len(X_val)):
        pe, std = ensemble_point_and_std(X_val[i], models, scaler, dropped_idx_list)
        point_ests[i] = pe
        stds[i] = std
        z_dists[i] = compute_ood_zscore_distance(X_val[i], scaler)
    bucket_ids = np.array([_bucket_index(v, bucket_edges) for v in y_val])

    def run_gate(std_threshold):
        """is_ood if z_dist > Z_THRESHOLD OR std > std_threshold. Returns per-bucket results."""
        is_ood_all = (z_dists > Z_THRESHOLD) | (stds > std_threshold)
        results = {}
        for b, edges in enumerate(BUCKETS):
            mask = bucket_ids == b
            n = int(mask.sum())
            if n == 0:
                results[edges] = (None, 0, 0)
                continue
            covered = 0
            idxs = np.where(mask)[0]
            for idx in idxs:
                y_cal, lb = apply_calibration(point_ests[idx], params, is_ood=bool(is_ood_all[idx]))
                if y_val[idx] >= lb:
                    covered += 1
            n_flagged = int(is_ood_all[idxs].sum())
            results[edges] = (covered / n, n, n_flagged)
        return results, int(is_ood_all.sum())

    # Baseline: current gate (z-distance only, std_threshold effectively infinite).
    print("\n" + "=" * 78)
    print("BASELINE (current gate: z_distance > 2.0 only)")
    print("=" * 78)
    baseline_results, baseline_total_flagged = run_gate(std_threshold=float("inf"))
    for edges, (cov, n, nf) in baseline_results.items():
        status = "PASS" if cov is not None and cov >= 0.9 else "FAIL"
        print(f"  bucket {edges}: coverage={cov:.1%}  n={n:4d}  flagged={nf:3d}  -> {status}")
    print(f"  TOTAL flagged: {baseline_total_flagged} / {len(X_val)}")

    # Sweep candidate std thresholds.
    for thr in STD_THRESHOLDS_TO_TRY:
        print("\n" + "=" * 78)
        print(f"OR-GATE CANDIDATE: z_distance > {Z_THRESHOLD} OR std > {thr}")
        print("=" * 78)
        results, total_flagged = run_gate(std_threshold=thr)
        all_pass = True
        for edges, (cov, n, nf) in results.items():
            status = "PASS" if cov is not None and cov >= 0.9 else "FAIL"
            all_pass = all_pass and (cov is not None and cov >= 0.9)
            delta_flagged = nf - dict(zip(BUCKETS, [v[2] for v in baseline_results.values()]))[edges]
            print(f"  bucket {edges}: coverage={cov:.1%}  n={n:4d}  flagged={nf:3d} "
                  f"(+{delta_flagged} vs baseline)  -> {status}")
        print(f"  TOTAL flagged: {total_flagged} / {len(X_val)} "
              f"(+{total_flagged - baseline_total_flagged} vs baseline)")
        print(f"  {'ALL BUCKETS PASS' if all_pass else 'AT LEAST ONE BUCKET STILL FAILS'}")

    print("\n" + "=" * 78)
    print("How to read this:")
    print("Pick the SMALLEST std threshold where bucket (0,50) passes AND no other")
    print("bucket's flagged-count jumps enough to drag its coverage below 90%. If no")
    print("threshold in this sweep achieves that, try values between 12 and 18, or")
    print("treat the near-miss as a documented model limitation (see doc section 4)")
    print("rather than continuing to tune the gate.")
    print("=" * 78)