"""
compare_raw_vs_calibrated_coverage.py -- before changing predict_rul_ensemble's
calibrated default based on check_per_bucket_coverage's verdict, this checks
whether RAW (uncalibrated) lower bounds actually do BETTER on the failing
(0,50) bucket -- or worse, in which case flipping the default to False would
be a regression, not a fix.

For raw (calibrated=False), "coverage" is defined the same way: fraction of
windows where true_rul >= lower_bound. Uses predict_rul_ensemble's own raw
lower_bound_timesteps (min across ensemble variants, clipped >= 0) -- NOT
apply_calibration, since we're checking calibration's counterfactual absence.

Reports coverage for ALL FOUR buckets, both raw and calibrated, so the
decision is based on the full picture, not just the one failing bucket.

Run: python models\rul\compare_raw_vs_calibrated_coverage.py   (from project root, AeroTwin/)
"""
import sys, os
sys.path.insert(0, 'models/rul')
import numpy as np
from sklearn.model_selection import train_test_split

from train_rul_ensemble import load_ensemble, load_all_flights, build_windowed_dataset_v5, predict_rul_ensemble
from calibrate_rul import load_calibration_params, apply_calibration, _bucket_index, BUCKET_EDGES

print("Loading ensemble ...")
models, scaler, dropped_idx_list = load_ensemble()
params = load_calibration_params()

print("Rebuilding test set (same split as run_coverage_check.py) ...")
flights = load_all_flights()
X, y, flight_ids = build_windowed_dataset_v5(flights)
unique_flights = np.unique(flight_ids)
train_flights, temp_flights = train_test_split(unique_flights, test_size=0.3, random_state=42)
val_flights, test_flights = train_test_split(temp_flights, test_size=0.5, random_state=42)
test_mask = np.isin(flight_ids, test_flights)
X_test, y_test = X[test_mask], y[test_mask]
y_test = np.asarray(y_test, dtype=float)

bucket_ids = np.array([_bucket_index(v, BUCKET_EDGES) for v in y_test])

raw_covered = {b: 0 for b in range(len(BUCKET_EDGES))}
cal_covered = {b: 0 for b in range(len(BUCKET_EDGES))}
bucket_n = {b: 0 for b in range(len(BUCKET_EDGES))}

# Also track mean lower_bound per bucket for both, so we can see WHY
# coverage differs, not just the pass/fail number.
raw_lb_sum = {b: 0.0 for b in range(len(BUCKET_EDGES))}
cal_lb_sum = {b: 0.0 for b in range(len(BUCKET_EDGES))}

print(f"Scanning {len(X_test)} test windows ...")
for idx in range(len(X_test)):
    b = bucket_ids[idx]
    bucket_n[b] += 1
    true_val = y_test[idx]

    raw_result = predict_rul_ensemble(X_test[idx], models, scaler, dropped_idx_list, calibrated=False)
    raw_lb = raw_result["rul_lower_bound_timesteps"]
    raw_point = raw_result["point_estimate_timesteps"]

    y_cal, cal_lb = apply_calibration(raw_point, params)

    raw_lb_sum[b] += raw_lb
    cal_lb_sum[b] += cal_lb

    if true_val >= raw_lb:
        raw_covered[b] += 1
    if true_val >= cal_lb:
        cal_covered[b] += 1

print(f"\n{'Bucket':<12}{'n':>6}{'Raw cov.':>12}{'Cal cov.':>12}{'Raw mean lb':>14}{'Cal mean lb':>14}")
for b, edges in enumerate(BUCKET_EDGES):
    n = bucket_n[b]
    if n == 0:
        print(f"{str(edges):<12}{n:>6}{'N/A':>12}{'N/A':>12}")
        continue
    raw_cov = raw_covered[b] / n
    cal_cov = cal_covered[b] / n
    raw_mean_lb = raw_lb_sum[b] / n
    cal_mean_lb = cal_lb_sum[b] / n
    print(f"{str(edges):<12}{n:>6}{raw_cov:>11.1%} {cal_cov:>11.1%} {raw_mean_lb:>13.2f} {cal_mean_lb:>13.2f}")

print("\nInterpretation:")
print("  If raw coverage on (0,50) >= 90% and beats calibrated -> flipping to")
print("    calibrated=False is a genuine fix for that bucket.")
print("  If raw coverage on (0,50) is ALSO below 90% (or worse than calibrated)")
print("    -> disabling calibration would not fix the (0,50) undercoverage, and")
print("    may make OTHER buckets worse (calibration was likely fixing real bias")
print("    elsewhere) -- re-tuning the (0,50) bucket's conformal band specifically")
print("    would be the better fix instead of a global default flip.")