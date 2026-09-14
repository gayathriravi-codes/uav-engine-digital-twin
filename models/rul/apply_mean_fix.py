"""
apply_mean_fix.py -- one-off script to make the sum->mean edit to
_compute_derived_features in train_rul_ensemble.py, without hand-editing
in Notepad (which has been unreliable this session).

Run: python models\rul\apply_mean_fix.py   (from project root, AeroTwin/)
"""
path = "models/rul/train_rul_ensemble.py"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old = """    vib_variance = np.var(window[:, vib_idx])

    return np.array([egt_slope, total_deviation, vib_variance])"""

new = """    # Changed from sum to mean across the window (divide by window.shape[0]).
    # A sum accumulates independent per-timestep noise linearly with window
    # length, causing this feature to explode under sensor noise even when
    # the noise itself is small/smooth -- confirmed via robustness_tests.py
    # Test 1 (both i.i.d. and correlated noise variants collapsed the RUL
    # point estimate to ~0 even at 1% noise). Mean keeps the feature on a
    # roughly noise-invariant scale regardless of window length.
    mean_deviation = total_deviation / window.shape[0]

    vib_variance = np.var(window[:, vib_idx])

    return np.array([egt_slope, mean_deviation, vib_variance])"""

count = content.count(old)
if count != 1:
    print(f"ABORTING: expected exactly 1 match for the target block, found {count}.")
    print("No changes made. File is untouched.")
else:
    new_content = content.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("SUCCESS: patch applied.")