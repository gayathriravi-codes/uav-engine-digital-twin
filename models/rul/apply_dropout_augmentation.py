"""
apply_dropout_augmentation.py -- adds per-window sensor dropout during
training (not just per-variant, which apply_feature_bagging already does).
This teaches the model that a single missing/zeroed sensor mid-flight is a
normal thing to see, rather than an unfamiliar pattern -- addressing
robustness_tests.py Test 2's large point-estimate swings on sensor dropout.

Safe by construction: only writes if it finds exactly one match for each
target block. Aborts with no changes otherwise.

Run: python models\rul\apply_dropout_augmentation.py   (from project root, AeroTwin/)
"""
path = "models/rul/train_rul_ensemble.py"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# ---------------------------------------------------------------------
# Patch 1: add the new apply_window_level_dropout() function right after
# apply_feature_bagging() (same file, same style, reuses SENSOR_FIELDS
# and _compute_derived_features already imported/defined above).
# ---------------------------------------------------------------------
old_1 = """    dropped_idx = np.array(sorted(dropped))
    X_bagged = X.copy()
    X_bagged[:, :, dropped_idx] = 0.0
    return X_bagged, dropped_idx"""

new_1 = """    dropped_idx = np.array(sorted(dropped))
    X_bagged = X.copy()
    X_bagged[:, :, dropped_idx] = 0.0
    return X_bagged, dropped_idx


def apply_window_level_dropout(X, drop_prob=0.15, seed=0):
    \"\"\"
    Per-window sensor dropout augmentation (training only).

    Unlike apply_feature_bagging (which zeroes the SAME sensor(s) for
    every window in a given variant, for ensemble diversity),
    this zeroes ONE randomly chosen raw sensor in a random SUBSET of
    individual windows, independently -- simulating what a real
    single-sensor dropout mid-flight would look like, at training time.

    X: np.array shape (n_windows, window_size, 10) -- featurized windows
       (7 raw sensors + 3 derived features), same shape as elsewhere in
       this file.
    drop_prob: fraction of windows that get one sensor zeroed. 0.15 is a
       starting point -- tuneable if val RMSE degrades too much.
    seed: RNG seed, offset from the variant's other seed uses so it
       doesn't correlate with bootstrap/feature-bagging randomness.

    After zeroing a raw sensor in an affected window, the 3 derived
    features (egt_slope, mean_deviation, vib_variance) are recomputed
    from the now-modified raw sensors, so the window stays internally
    consistent -- same fix pattern used in robustness_tests.py's tests.

    Returns: X_augmented, same shape as X.
    \"\"\"
    rng = np.random.RandomState(seed + 5000)  # offset from other seed uses
    X_aug = X.copy()
    n_windows = X_aug.shape[0]

    affected = rng.random(n_windows) < drop_prob
    affected_idx = np.where(affected)[0]

    for i in affected_idx:
        sensor_idx = rng.randint(0, len(SENSOR_FIELDS))
        X_aug[i, :, sensor_idx] = 0.0
        new_derived = _compute_derived_features(X_aug[i, :, :len(SENSOR_FIELDS)])
        X_aug[i, :, len(SENSOR_FIELDS):] = new_derived

    return X_aug, affected_idx"""

count_1 = content.count(old_1)

# ---------------------------------------------------------------------
# Patch 2: call apply_window_level_dropout() on the bootstrapped training
# data inside train_one_variant, AFTER apply_feature_bagging (variant-level
# dropping), BEFORE building the DataLoader. Eval data (X_eval) is left
# untouched -- augmentation is training-time only, same as bootstrapping
# and feature bagging already are.
# ---------------------------------------------------------------------
old_2 = """    X_train_boot, y_train_boot = bootstrap_by_flight(X_train, y_train, train_flight_ids, seed)
    X_train_boot, dropped_idx = apply_feature_bagging(X_train_boot, n_dropped_features, seed)
    X_eval_bagged = X_eval.copy()"""

new_2 = """    X_train_boot, y_train_boot = bootstrap_by_flight(X_train, y_train, train_flight_ids, seed)
    X_train_boot, dropped_idx = apply_feature_bagging(X_train_boot, n_dropped_features, seed)
    # Per-window dropout augmentation (training only) -- teaches the model
    # that a single missing/zeroed sensor mid-flight is a normal pattern,
    # not something to panic over. Addresses robustness_tests.py Test 2's
    # large point-estimate swings on single-sensor dropout.
    X_train_boot, _ = apply_window_level_dropout(X_train_boot, drop_prob=0.15, seed=seed)
    X_eval_bagged = X_eval.copy()"""

count_2 = content.count(old_2)

if count_1 != 1 or count_2 != 1:
    print(f"ABORTING: expected exactly 1 match each. Found patch_1={count_1}, patch_2={count_2}.")
    print("No changes made. File is untouched.")
else:
    new_content = content.replace(old_1, new_1).replace(old_2, new_2)
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("SUCCESS: both patches applied.")
    print(" - apply_window_level_dropout() added after apply_feature_bagging()")
    print(" - train_one_variant() now calls it on X_train_boot before building the DataLoader")