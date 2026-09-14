"""
raise_ood_floor.py -- bumps ood_std_floor default so the gate's inflated
std is actually higher than typical baseline std (34-35 in our tests),
not just higher than the raw OOD-input ensemble std. Otherwise the gate
fires but produces a LOWER number than baseline, still reading as
overconfident even though the mechanism worked.

Run: python models\rul\raise_ood_floor.py   (from project root, AeroTwin/)
"""
path = "models/rul/train_rul_ensemble.py"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old = """def predict_rul_ensemble(window, models, scaler, dropped_idx_list, calibrated=True,
                          ood_gate=True, ood_zscore_threshold=2.0, ood_std_floor=25.0):"""

new = """def predict_rul_ensemble(window, models, scaler, dropped_idx_list, calibrated=True,
                          ood_gate=True, ood_zscore_threshold=2.0, ood_std_floor=50.0):"""

count = content.count(old)
if count != 1:
    print(f"ABORTING: expected exactly 1 match, found {count}. No changes made.")
else:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.replace(old, new))
    print("SUCCESS: ood_std_floor default raised from 25.0 to 50.0")