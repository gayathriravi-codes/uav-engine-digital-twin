"""
calibrate_rul.py -- Step B: per-bucket bias correction + conformal
calibration for AeroTwin's RUL ensemble.

Separate file from train_rul_ensemble.py on purpose: calibration is a
post-processing step with its own inputs (validation predictions/labels),
its own saved artifact (per-bucket bias + conformal width), and its own
fit/apply lifecycle. Keeping it out of train_rul_ensemble.py avoids mixing
training concerns with calibration concerns.

Design (agreed, from Step B plan):
  1. Bias correction (fixes P1, per bucket):
       r_i = y_true_i - point_estimate_i      (SIGNED, not absolute)
       corrected_pred = point_estimate - mean_residual_bucket
  2. Per-bucket conformal band around the CORRECTED estimate (fixes P2):
       lb = corrected_pred - quantile_0.9(|y_true - corrected_pred| in bucket)

This guarantees lb <= corrected_pred always. Acceptance test is per-bucket
coverage (fraction of windows where y_true >= lb) on the TEST set, not an
overall average -- see check_per_bucket_coverage().

Buckets (current, after merging (250,350) into (200,250) on 2026-09-13 --
confirmed via check_bucket_frequency.py that (250,350) never gets
populated by any real or what-if-generated window, 0/29,268 + 0/120):
    [(0,50), (50,100), (100,200), (200,350)]

Usage from train_rul_ensemble.py's predict_rul_ensemble(calibrated=True):
    from calibrate_rul import apply_calibration, load_calibration_params
    params = load_calibration_params()
    y_cal, lb = apply_calibration(point_estimate, params)
"""
import os
import numpy as np
import joblib


from train_rul import MODEL_OUT_DIR

BUCKET_EDGES = [(0, 50), (50, 100), (100, 200), (200, 350)]
CONFORMAL_QUANTILE = 0.9
CALIBRATION_PATH = os.path.join(MODEL_OUT_DIR, "rul_calibration.joblib")


def _bucket_index(value, bucket_edges=BUCKET_EDGES):
    """Bucket index for a single RUL value. Clamps to the nearest bucket
    at the extremes instead of falling through."""
    for i, (lo, hi) in enumerate(bucket_edges):
        if lo <= value < hi:
            return i
    return 0 if value < bucket_edges[0][0] else len(bucket_edges) - 1


def fit_calibration(X_val, y_val, models, scaler, dropped_idx_list,
                     bucket_edges=BUCKET_EDGES, conformal_quantile=CONFORMAL_QUANTILE):
    """
    Two-stage fit, to match how apply_calibration must look up buckets at
    inference (where true RUL is unknown):

      Stage 0 (rough): bucket by TRUE RUL (clean, no contamination -- this
        is what the original single-stage fit did). Gives a rough bias per
        bucket, used only to nudge the raw point estimate closer to true
        before the real bucket assignment.

      Stage 1 (final): apply stage-0's rough bias to every validation point,
        then RE-BUCKET by that corrected estimate (not by true value). This
        matches what apply_calibration will do at inference (it has no true
        value either, only a corrected estimate). Compute the FINAL bias +
        conformal width within these re-bucketed groups.

    This is the "iterate once" fallback: a single rough correction wasn't
    enough to move badly-biased windows across a bucket boundary, so we
    fit the final numbers against the bucket assignment inference will
    actually use, rather than against the (unreachable at inference) true
    bucket.

    Saves BOTH stage-0 ("rough_buckets") and stage-1 ("buckets") params, so
    apply_calibration can do the same two-pass lookup at inference.
    """
    from train_rul_ensemble import predict_rul_ensemble

    point_estimates = np.array([
        predict_rul_ensemble(X_val[i], models, scaler, dropped_idx_list)["point_estimate_timesteps"]
        for i in range(len(X_val))
    ])
    y_val = np.asarray(y_val, dtype=float)

    # --- Stage 0: rough bias, bucketed by TRUE RUL (clean) ---
    true_bucket_ids = np.array([_bucket_index(v, bucket_edges) for v in y_val])
    rough_buckets = {}
    for b, edges in enumerate(bucket_edges):
        mask = true_bucket_ids == b
        n = int(mask.sum())
        if n == 0:
            rough_buckets[b] = {"mean_residual": 0.0, "n": 0}
            continue
        residuals = y_val[mask] - point_estimates[mask]
        rough_buckets[b] = {"mean_residual": float(np.mean(residuals)), "n": n}

    # Apply stage-0 rough bias to every validation point.
    rough_corrected = np.array([
        point_estimates[i] + rough_buckets[_bucket_index(point_estimates[i], bucket_edges)]["mean_residual"]
        for i in range(len(point_estimates))
    ])

    # --- Stage 1: final bias + conformal width, bucketed by the CORRECTED
    # estimate -- this is the bucket assignment inference will actually use.
    corrected_bucket_ids = np.array([_bucket_index(v, bucket_edges) for v in rough_corrected])
    buckets = {}
    for b, edges in enumerate(bucket_edges):
        mask = corrected_bucket_ids == b
        n = int(mask.sum())
        if n == 0:
            print(f"[calibrate_rul] WARNING: bucket {edges} has 0 validation "
                  f"windows after re-bucketing by corrected estimate -- "
                  f"bias=0, band=0, do not trust this bucket.")
            buckets[b] = {"mean_residual": 0.0, "conformal_width": 0.0, "n": 0}
            continue

        preds_b = point_estimates[mask]
        true_b = y_val[mask]

        residuals = true_b - preds_b
        mean_residual = float(np.mean(residuals))

        corrected_b = preds_b + mean_residual
        corrected_abs_resid = np.abs(true_b - corrected_b)
        conformal_width = float(np.quantile(corrected_abs_resid, conformal_quantile))

        buckets[b] = {"mean_residual": mean_residual, "conformal_width": conformal_width, "n": n}

    params = {
        "bucket_edges": bucket_edges,
        "conformal_quantile": conformal_quantile,
        "rough_buckets": rough_buckets,
        "buckets": buckets,
    }

    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    joblib.dump(params, CALIBRATION_PATH)

    print(f"[calibrate_rul] Saved calibration params to {CALIBRATION_PATH}")
    print("  Stage 0 (rough, bucketed by true RUL):")
    for b, edges in enumerate(bucket_edges):
        info = rough_buckets[b]
        print(f"    bucket {edges}: n={info['n']:4d}  bias={info['mean_residual']:7.2f}")
    print("  Stage 1 (final, bucketed by corrected estimate -- matches inference):")
    for b, edges in enumerate(bucket_edges):
        info = buckets[b]
        print(f"    bucket {edges}: n={info['n']:4d}  bias={info['mean_residual']:7.2f}  "
              f"band={info['conformal_width']:7.2f}")

    return params

def load_calibration_params(path=CALIBRATION_PATH):
    return joblib.load(path)


def apply_calibration(point_estimate, params):
    """
    Two-pass lookup matching the two-stage fit above:
      pass 1: bucket the raw point estimate using STAGE-0 (rough_buckets,
              fit against true RUL) -- gives a rough corrected estimate.
      pass 2: bucket the rough-corrected estimate using STAGE-1 (buckets,
              fit against corrected-estimate buckets) -- this is the SAME
              bucket assignment rule fit_calibration used to compute the
              final numbers, so no train/inference mismatch.

    Returns (y_cal, lb): lb guaranteed <= y_cal, and >= 0.
    """
    bucket_edges = params["bucket_edges"]
    rough_buckets = params["rough_buckets"]
    buckets = params["buckets"]

    # Pass 1: rough correction (stage 0).
    b0 = _bucket_index(point_estimate, bucket_edges)
    bias0 = rough_buckets.get(b0, {"mean_residual": 0.0})["mean_residual"]
    rough_corrected = point_estimate + bias0

    # Pass 2: final bucket + correction (stage 1), matching how fit_calibration
    # assigned buckets during training.
    b1 = _bucket_index(rough_corrected, bucket_edges)
    info = buckets.get(b1, {"mean_residual": 0.0, "conformal_width": 0.0})

    y_cal = point_estimate + info["mean_residual"]
    lb = y_cal - info["conformal_width"]

    lb = min(lb, y_cal)
    lb = max(lb, 0.0)

    return float(y_cal), float(lb)
def check_per_bucket_coverage(X_test, y_test, models, scaler, dropped_idx_list,
                               params, target_coverage=0.9):
    """
    Acceptance test -- run ONCE, on the untouched test set, after Step B is
    implemented. Checks fraction of windows where y_true >= lb, PER BUCKET,
    not an overall average (a bucket individually failing 90% could hide
    behind a fine-looking overall number).

    Also spot-checks: lb <= y_cal always, and lb never negative.

    Returns dict: {bucket_edges: {"coverage": float, "n": int, "pass": bool}}
    """
    from train_rul_ensemble import predict_rul_ensemble

    bucket_edges = params["bucket_edges"]
    y_test = np.asarray(y_test, dtype=float)
    bucket_ids = np.array([_bucket_index(y, bucket_edges) for y in y_test])

    results = {}
    for b, edges in enumerate(bucket_edges):
        mask = bucket_ids == b
        n = int(mask.sum())
        if n == 0:
            results[edges] = {"coverage": None, "n": 0, "pass": False}
            continue

        covered = 0
        idxs = np.where(mask)[0]
        for idx in idxs:
            point_estimate = predict_rul_ensemble(
                X_test[idx], models, scaler, dropped_idx_list
            )["point_estimate_timesteps"]
            y_cal, lb = apply_calibration(point_estimate, params)

            assert lb <= y_cal + 1e-6, "Lower bound exceeded corrected point estimate!"
            assert lb >= 0, "Lower bound went negative!"

            if y_test[idx] >= lb:
                covered += 1

        coverage = covered / n
        results[edges] = {"coverage": coverage, "n": n, "pass": coverage >= target_coverage}

    print(f"[calibrate_rul] Per-bucket coverage on TEST (target >= {target_coverage:.0%}):")
    all_pass = True
    for edges, r in results.items():
        cov_str = f"{r['coverage']:.1%}" if r["coverage"] is not None else "N/A (n=0)"
        status = "PASS" if r["pass"] else "FAIL"
        print(f"  bucket {edges}: coverage={cov_str}  n={r['n']:4d}  -> {status}")
        all_pass = all_pass and r["pass"]
    print(f"[calibrate_rul] {'ALL BUCKETS PASS -- safe to flip calibrated=True default' if all_pass else 'AT LEAST ONE BUCKET FAILED -- keep calibrated=False default'}")

    return results


def predict_rul_calibrated(window, models, scaler, dropped_idx_list):
    """
    Convenience: full path from a featurized window to a calibrated
    prediction in one call. Chains predict_rul_ensemble(calibrated=False)
    -> apply_calibration(...).

    Local import of predict_rul_ensemble avoids a circular import, since
    train_rul_ensemble.py optionally imports apply_calibration from this
    file when calibrated=True is passed.
    """
    from train_rul_ensemble import predict_rul_ensemble

    raw = predict_rul_ensemble(window, models, scaler, dropped_idx_list, calibrated=False)
    params = load_calibration_params()
    y_cal, lb = apply_calibration(raw["point_estimate_timesteps"], params)

    return {
        "point_estimate_timesteps": y_cal,
        "rul_lower_bound_timesteps": lb,
        "std_timesteps": raw["std_timesteps"],
    }