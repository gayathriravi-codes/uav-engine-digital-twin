import re

FILES = [
    "check_vibration.py",
    "check_vibration_fixed.py",
    r"models\classification\engine_inference.py",
    r"models\rul\calibrate_rul.py",
    r"models\rul\check_bucket_frequency.py",
    r"models\rul\diagnose_250_350_bucket.py",
    r"models\rul\diagnose_underestimation.py",
    r"models\rul\robustness_tests.py",
    r"models\rul\sanity_check_calibration.py",
    r"models\rul\train_rul_ensemble.py",
    r"models\rul\train_rul_ensemble_v6_experimental.py",
    r"models\rul\whatif_engine.py",
    "test_whatif_all_faults.py",
]

RENAMES = {
    "point_estimate_minutes": "point_estimate_timesteps",
    "rul_lower_bound_minutes": "rul_lower_bound_timesteps",
    "std_minutes": "std_timesteps",
}

for path in FILES:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    original = content
    for old, new in RENAMES.items():
        # word-boundary match so e.g. delta_minutes is untouched
        content = re.sub(rf"\b{old}\b", new, content)
    if content != original:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Updated: {path}")
    else:
        print(f"No change: {path}")