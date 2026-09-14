"""
fix_negative_ycal.py -- fixes calibrate_rul.py's apply_calibration(): it
clamped lb >= 0 but never clamped y_cal >= 0, so a large negative bias
correction on a near-zero true-RUL window could push y_cal negative while
lb got floored at 0, inverting lb > y_cal. Found via debug_coverage_failure.py
on test idx=54 (true_rul=0.0, point_estimate=1.64, y_cal=-1.33, lb=0.0).

Run: python models\rul\fix_negative_ycal.py   (from project root, AeroTwin/)
"""
path = "models/rul/calibrate_rul.py"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old = """    y_cal = point_estimate + info["mean_residual"]
    lb = y_cal - info["conformal_width"]

    lb = min(lb, y_cal)
    lb = max(lb, 0.0)

    return float(y_cal), float(lb)"""

new = """    y_cal = point_estimate + info["mean_residual"]
    # Clamp y_cal to >= 0 BEFORE computing lb. A large negative bias
    # correction (e.g. on a near-zero true-RUL window) can otherwise push
    # y_cal negative while lb gets floored at 0 below, inverting the
    # lb <= y_cal invariant (found via check_per_bucket_coverage on the
    # test set: true_rul=0.0, y_cal went to -1.33 while lb floored to 0).
    y_cal = max(y_cal, 0.0)
    lb = y_cal - info["conformal_width"]

    lb = min(lb, y_cal)
    lb = max(lb, 0.0)

    return float(y_cal), float(lb)"""

count = content.count(old)
if count != 1:
    print(f"ABORTING: expected exactly 1 match, found {count}. No changes made.")
else:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.replace(old, new))
    print("SUCCESS: apply_calibration now clamps y_cal >= 0 before computing lb.")