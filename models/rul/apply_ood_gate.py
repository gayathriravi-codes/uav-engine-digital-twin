"""
apply_ood_gate.py -- adds a post-hoc OOD gate to predict_rul_ensemble().
Computes the input window's average per-feature z-score distance from the
training distribution (reusing the already-fit StandardScaler -- no new
fitting step, no matrix inversion, no retrain). If the distance exceeds a
threshold, std is inflated to reflect genuine uncertainty on unfamiliar
inputs. Addresses robustness_tests.py Test 4: ensemble std DECREASING on
out-of-distribution combined-fault inputs.

Safe by construction: only writes if it finds exactly one match for each
target block. Aborts with no changes otherwise.

Run: python models\rul\apply_ood_gate.py   (from project root, AeroTwin/)
"""
path = "models/rul/train_rul_ensemble.py"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# ---------------------------------------------------------------------
# Patch 1: add compute_ood_zscore_distance() right before predict_rul_ensemble().
# ---------------------------------------------------------------------
old_1 = """def predict_rul_ensemble(window, models, scaler, dropped_idx_list, calibrated=True):"""

new_1 = """def compute_ood_zscore_distance(window, scaler):
    \"\"\"
    Post-hoc OOD signal: average per-feature |z-score| of this window's
    values relative to the TRAINING distribution, using the already-fit
    StandardScaler's mean_ and scale_ (no new fitting step, no covariance
    matrix, no retrain -- avoids the numerical risk of inverting a
    possibly near-singular covariance matrix on a modest dataset).

    window: np.array shape (window_size, 10) -- the FEATURIZED window,
        same shape predict_rul_ensemble() expects.
    scaler: the already-fit StandardScaler (has .mean_ and .scale_,
        length 10, one per feature).

    Returns: float -- mean absolute z-score across all timesteps and
        features. Higher = further from what the model was trained on.
    \"\"\"
    z = (window - scaler.mean_) / scaler.scale_
    return float(np.abs(z).mean())


def predict_rul_ensemble(window, models, scaler, dropped_idx_list, calibrated=True,
                          ood_gate=True, ood_zscore_threshold=2.0, ood_std_floor=25.0):"""

count_1 = content.count(old_1)

# ---------------------------------------------------------------------
# Patch 2: apply the gate right before the return statements (both the
# calibrated and non-calibrated paths need it, so it goes after std is
# computed but is shared by both branches).
# ---------------------------------------------------------------------
old_2 = """    preds = np.array(preds)
    point_estimate = max(0.0, float(preds.mean()))
    lower_bound = max(0.0, float(preds.min()))
    std = float(preds.std())

    if calibrated:"""

new_2 = """    preds = np.array(preds)
    point_estimate = max(0.0, float(preds.mean()))
    lower_bound = max(0.0, float(preds.min()))
    std = float(preds.std())

    # Post-hoc OOD gate: if this window's average feature z-score distance
    # from the TRAINING distribution exceeds ood_zscore_threshold, inflate
    # std to reflect genuine uncertainty on an unfamiliar input, rather
    # than reporting the ensemble's (possibly overconfident) raw spread.
    # This does NOT change point_estimate or lower_bound -- only std, so
    # existing callers relying on point_estimate/lower_bound are unaffected.
    if ood_gate:
        ood_distance = compute_ood_zscore_distance(window, scaler)
        if ood_distance > ood_zscore_threshold:
            std = max(std, ood_std_floor)

    if calibrated:"""

count_2 = content.count(old_2)

if count_1 != 1 or count_2 != 1:
    print(f"ABORTING: expected exactly 1 match each. Found patch_1={count_1}, patch_2={count_2}.")
    print("No changes made. File is untouched.")
else:
    new_content = content.replace(old_1, new_1).replace(old_2, new_2)
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("SUCCESS: both patches applied.")
    print(" - compute_ood_zscore_distance() added before predict_rul_ensemble()")
    print(" - predict_rul_ensemble() now gates std based on OOD distance (new kwargs: ood_gate=True, ood_zscore_threshold=2.0, ood_std_floor=25.0)")