"""
fix_double_calibration.py -- fixes a bug in check_per_bucket_coverage():
it called predict_rul_ensemble() with no calibrated= argument, which
defaults to True (per Step B), then called apply_calibration() on that
ALREADY-calibrated value -- double-applying bias correction and rebucketing
an already-shifted estimate. This caused the "lb > y_cal" assertion
failure. Fix: request the raw (uncalibrated) point estimate, since
apply_calibration() is what's meant to calibrate it.

Pre-existing bug, unrelated to tonight's other changes -- surfaced only
now because this was the check's first-ever run.

Run: python models\\rul\\fix_double_calibration.py   (from project root, AeroTwin/)
"""
path = "models/rul/calibrate_rul.py"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old = """            point_estimate = predict_rul_ensemble(
                X_test[idx], models, scaler, dropped_idx_list
            )["point_estimate_timesteps"]
            y_cal, lb = apply_calibration(point_estimate, params)"""

new = """            # calibrated=False: this coverage check calibrates the point
            # estimate ITSELF via apply_calibration() below. Calling
            # predict_rul_ensemble() with no calibrated= arg (default True)
            # would double-calibrate -- bias-correcting an already
            # bias-corrected value and rebucketing an already-shifted
            # estimate, which can violate lb <= y_cal.
            point_estimate = predict_rul_ensemble(
                X_test[idx], models, scaler, dropped_idx_list, calibrated=False
            )["point_estimate_timesteps"]
            y_cal, lb = apply_calibration(point_estimate, params)"""

count = content.count(old)
if count != 1:
    print(f"ABORTING: expected exactly 1 match, found {count}. No changes made.")
else:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.replace(old, new))
    print("SUCCESS: check_per_bucket_coverage now requests calibrated=False from predict_rul_ensemble.")