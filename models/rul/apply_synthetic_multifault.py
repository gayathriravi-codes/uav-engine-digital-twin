"""
apply_synthetic_multifault.py -- adds synthetic multi-fault training
windows by combining pairs of existing single-fault windows' sensor
deviations. Addresses robustness_tests.py Test 4: ensemble std DECREASING
(overconfidence) on out-of-distribution combined-fault inputs, because the
model has never seen any combined-fault pattern during training.

Safe by construction: only writes if it finds exactly one match for each
target block. Aborts with no changes otherwise.

Run: python models\rul\apply_synthetic_multifault.py   (from project root, AeroTwin/)
"""
path = "models/rul/train_rul_ensemble.py"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# ---------------------------------------------------------------------
# Patch 1: add generate_synthetic_multifault_windows() right after
# build_windowed_dataset_v5().
# ---------------------------------------------------------------------
old_1 = """    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y, np.array(all_flight_ids)"""

new_1 = """    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y, np.array(all_flight_ids)


def generate_synthetic_multifault_windows(X, y, n_synthetic, seed=0):
    \"\"\"
    Builds synthetic multi-fault windows by combining pairs of existing
    (real, single-fault) windows' RAW sensor deviations from healthy
    midpoints, added together onto one of the two base windows. This
    approximates what a genuine combined-fault signature would look like,
    without needing new simulator data.

    Deliberately mirrors robustness_tests.py Test 4's own construction
    (combining an oil-issue-like and vibration-fault-like signature in one
    window) -- so the model sees SOME combined-fault variance during
    training, rather than encountering it for the first time at test time.

    X: np.array shape (n_windows, window_size, 10) -- featurized windows.
    y: np.array shape (n_windows,) -- true RUL labels.
    n_synthetic: how many synthetic windows to generate.
    seed: RNG seed, offset from other seed uses.

    The label for each synthetic window is the MIN of the two source
    windows' labels (conservative -- a combined fault should not be
    assumed less urgent than either fault alone).

    Derived features (egt_slope, mean_deviation, vib_variance) are
    recomputed from the combined raw sensors so each synthetic window is
    internally consistent.

    Returns: X_synthetic, y_synthetic -- to be concatenated onto the real
    training set (NOT val/test -- synthetic augmentation is train-only,
    same principle as bootstrapping/feature bagging/window dropout).
    \"\"\"
    rng = np.random.RandomState(seed + 9000)  # offset from other seed uses
    n_real = X.shape[0]
    n_raw_sensors = len(SENSOR_FIELDS)

    idx_a = rng.randint(0, n_real, size=n_synthetic)
    idx_b = rng.randint(0, n_real, size=n_synthetic)

    X_synthetic = X[idx_a].copy()
    y_synthetic = np.minimum(y[idx_a], y[idx_b])

    for k in range(n_synthetic):
        base = X[idx_a[k], :, :n_raw_sensors]
        other = X[idx_b[k], :, :n_raw_sensors]
        # Combine deviations from each sensor's healthy midpoint, rather
        # than raw values, so the combination reflects "both fault
        # signatures present" rather than an arbitrary blend.
        combined_raw = base.copy()
        for s_idx, sensor in enumerate(SENSOR_FIELDS):
            lo, hi = HEALTHY_RANGES[sensor]
            midpoint = (lo + hi) / 2
            base_dev = base[:, s_idx] - midpoint
            other_dev = other[:, s_idx] - midpoint
            combined_raw[:, s_idx] = midpoint + base_dev + other_dev

        new_derived = _compute_derived_features(combined_raw)
        X_synthetic[k, :, :n_raw_sensors] = combined_raw
        X_synthetic[k, :, n_raw_sensors:] = new_derived

    return X_synthetic, y_synthetic"""

count_1 = content.count(old_1)

# ---------------------------------------------------------------------
# Patch 2: in __main__, after building X_train/y_train (post train/val/test
# split), append synthetic multi-fault windows to the TRAIN set only.
# ---------------------------------------------------------------------
old_2 = """    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    train_flight_ids = flight_ids[train_mask]"""

new_2 = """    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    train_flight_ids = flight_ids[train_mask]

    # Synthetic multi-fault augmentation (train-only, ~10% of real train
    # windows) -- addresses Test 4's OOD overconfidence by giving the
    # model SOME combined-fault variance to learn from. Synthetic windows
    # get a placeholder flight_id so bootstrap_by_flight treats them as
    # their own single-window "flights" (sampled independently, not
    # tied to any real flight's other windows).
    n_synthetic = int(0.10 * len(X_train))
    print(f"Generating {n_synthetic} synthetic multi-fault training windows ...")
    X_synth, y_synth = generate_synthetic_multifault_windows(X_train, y_train, n_synthetic, seed=42)
    synth_flight_ids = np.array([f"synthetic_{i}" for i in range(n_synthetic)])
    X_train = np.concatenate([X_train, X_synth], axis=0)
    y_train = np.concatenate([y_train, y_synth], axis=0)
    train_flight_ids = np.concatenate([train_flight_ids, synth_flight_ids], axis=0)
    print(f"Train windows after synthetic augmentation: {len(X_train)}")"""

count_2 = content.count(old_2)

if count_1 != 1 or count_2 != 1:
    print(f"ABORTING: expected exactly 1 match each. Found patch_1={count_1}, patch_2={count_2}.")
    print("No changes made. File is untouched.")
else:
    new_content = content.replace(old_1, new_1).replace(old_2, new_2)
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("SUCCESS: both patches applied.")
    print(" - generate_synthetic_multifault_windows() added after build_windowed_dataset_v5()")
    print(" - __main__ now appends synthetic windows to X_train/y_train/train_flight_ids")