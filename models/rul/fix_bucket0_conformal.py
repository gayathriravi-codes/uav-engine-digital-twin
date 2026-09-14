"""
fix_bucket0_conformal.py -- gives bucket (0,50) its own (wider) conformal
quantile instead of sharing the global CONFORMAL_QUANTILE with every other
bucket.

Why: compare_raw_vs_calibrated_coverage.py showed calibrated coverage on
(0,50) at 87.1% (target >=90%), while every other bucket cleared 96%+.
Raw (uncalibrated) coverage was WORSE everywhere (61.8% on (0,50) alone),
so flipping the global calibrated=False default was ruled out -- the fix
is to widen just this bucket's band, not touch the others.

Patches models\\rul\\calibrate_rul.py:
  1. Inserts a BUCKET_QUANTILE_OVERRIDES dict (bucket 0 -> 0.94) right
     after `buckets = {}` inside fit_calibration.
  2. Changes the np.quantile(...) call to look up this bucket's quantile
     instead of always using the global conformal_quantile.

Exact-match-or-abort: if either anchor string isn't found verbatim, this
prints an error and changes nothing. Run from project root:
    python models\\rul\\fix_bucket0_conformal.py
"""

import shutil
import datetime

PATH = r"models\rul\calibrate_rul.py"

with open(PATH, "r", encoding="utf-8") as f:
    content = f.read()

# --- Patch 1: insert the override dict right after `buckets = {}` ---
anchor1 = "    buckets = {}\n"
insert1 = (
    "    buckets = {}\n"
    "    # Bucket-specific conformal quantile override: (0,50) undercovered\n"
    "    # (87.1% vs target 90%) at the global CONFORMAL_QUANTILE, so it gets\n"
    "    # a higher quantile -> wider band, while other buckets keep the\n"
    "    # global default. See fix_bucket0_conformal.py for the numbers.\n"
    "    BUCKET_QUANTILE_OVERRIDES = {0: 0.94}\n"
)

count1 = content.count(anchor1)
if count1 != 1:
    print(f"[fix_bucket0_conformal] ABORT: expected exactly 1 match for "
          f"anchor1, found {count1}. No changes made.")
    raise SystemExit(1)

# --- Patch 2: use the override when computing conformal_width ---
anchor2 = (
    '        conformal_width = float(np.quantile(corrected_abs_resid, conformal_quantile))\n'
)
replacement2 = (
    '        bucket_quantile = BUCKET_QUANTILE_OVERRIDES.get(b, conformal_quantile)\n'
    '        conformal_width = float(np.quantile(corrected_abs_resid, bucket_quantile))\n'
)

count2 = content.count(anchor2)
if count2 != 1:
    print(f"[fix_bucket0_conformal] ABORT: expected exactly 1 match for "
          f"anchor2, found {count2}. No changes made.")
    raise SystemExit(1)

# --- Backup before writing ---
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = PATH + f".bak_{stamp}"
shutil.copy2(PATH, backup_path)
print(f"[fix_bucket0_conformal] Backed up original to {backup_path}")

new_content = content.replace(anchor1, insert1, 1)
new_content = new_content.replace(anchor2, replacement2, 1)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(new_content)

print("[fix_bucket0_conformal] Patch applied successfully.")
print("[fix_bucket0_conformal] Next: re-run run_coverage_check.py (or your")
print("  compare_raw_vs_calibrated_coverage.py) to confirm (0,50) now")
print("  clears 90% and the other buckets didn't regress.")