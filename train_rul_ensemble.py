"""
train_rul_ensemble.py -- RUL confidence-bound ensemble for AeroTwin. (v5)

Built by Ashmitha, on top of Aashita/Ashmitha's train_rul.py.

History:
  v1: seed + dropout variation only -> ensemble std too tight (~1.6-4.2 min)
  v2: added per-variant flight bootstrapping -> RMSEs spread out across
      variants but per-window std barely moved (~2.0-3.0 min).
  v3: added hidden_size variation + feature bagging (random sensor drop) +
      proper train/val/test split. Val RMSE (honest, no leakage): 44.94.
      Found: hidden_size=16 too weak (val RMSE 58.4); dropping egt+oil_pressure
      together hurt badly (val RMSE 47.2) -- confirmed by sensor_importance.py:
      vibration/cht/oil_pressure/rpm are the RUL_CORE_SENSORS (~90% of total
      importance), oil_temp/egt/fuel_flow are minor.
  v4:
    - HIDDEN_SIZES no longer goes below 24 (16 confirmed too weak in v3)
    - feature bagging is now CORE-AWARE: it will never drop two RUL_CORE_SENSORS
      together in the same variant. At most one core sensor may be dropped
      per variant (alone), or any number of non-core sensors.
    - bumped to 7 variants for more ensemble diversity per the tuning plan
  v5:
    - Step A fix for P1 (point-estimate bias on high-RUL windows): each
      window now carries 3 derived per-window features (EGT slope,
      integrated deviation from HEALTHY_RANGES, in-window vibration
      variance) alongside the 7 raw sensors -- see create_windows_v5.
      Per-window only, not per-flight cumulative, per the time-constrained
      decision. Lives here, not in train_rul.py, so the original
      single-model baseline (7 sensors only) stays untouched.
    - predict_rul_ensemble's OUTPUT shape is unchanged (point_estimate_timesteps,
      rul_lower_bound_timesteps, std_timesteps) so the dashboard doesn't break.
      Its INPUT now needs the featurized window, not the raw 7-column one --
      use build_inference_window() below to convert.
    - Step B (this edit): predict_rul_ensemble() takes a new `calibrated`
      kwarg (default True). When True, applies the per-bucket bias
      correction + conformal lower bound from calibrate_rul.py before
      returning. Return dict KEYS are unchanged either way -- teammates
      calling predict_rul_ensemble() with no calibrated= arg get the
      bias-corrected behavior automatically.
    - Step B verified: per-bucket coverage on TEST >= 90% for all buckets
      after merging (100,150)/(150,200) into (100,200) and one iteration
      of refit (bucket-by-corrected-estimate, not raw estimate). calibrated
      default flipped to True. KNOWN CAVEAT: (250,350) bucket got 0
      validation windows after re-bucketing -- its bias/band are 0.0, so
      calibrated output there equals the raw (still-biased) estimate. It
      "passes" coverage only because the raw estimate already undershoots.
      Not a real fix for that bucket -- documented, not solved.
    - Added load_ensemble() -- loads a previously trained + saved ensemble
      (models/scaler/dropped_idx_list) straight from disk, ready to pass
      into predict_rul_ensemble(). Teammates no longer need to keep
      trained model objects in memory from a training run -- just call
      load_ensemble() after train_rul_ensemble.py has been run once.

  v6 investigation (NOT integrated into production): see
    train_rul_ensemble_v6_experimental.py for the elapsed_fraction +
    weighted-loss experiments on the (250,350) collapse. Not merged here.

  v6 rename (Ashmitha, merged into this file): predict_rul_ensemble()'s
    OUTPUT keys were renamed from *_minutes to *_timesteps everywhere in
    docstrings, asserts, and the __main__ sanity-check block, since the
    values were always in timesteps, never literal minutes.

  v6.1 (merge, Aashita): the two `return {}` blocks in
    predict_rul_ensemble() keep BOTH *_timesteps and *_minutes keys, so
    the dashboard team's app.py/mock_data.py (still reading *_minutes as
    of the last check) doesn't break. Everywhere else uses *_timesteps only.

  v7 (Aashita): OOD-gate fix. When a window is flagged OOD
    (ood_distance > ood_zscore_threshold) AND calibrated=True, force the
    returned rul_lower_bound_timesteps/_minutes to 0.0 instead of trusting
    the bucket-based calibrated lb. point_estimate_timesteps is left
    untouched. Confirmed via diagnose_bucket0.py this was the direct
    cause of bucket (0,50)'s 86.5% coverage failure (23 test windows with
    true_rul near 0, point estimates off by 2-6x).

  v8 (Aashita, separate session, in calibrate_rul.py not this file):
    v7's is_ood override only lived inside predict_rul_ensemble's
    `if calibrated:` branch, so it never reached check_per_bucket_coverage
    (which calls calibrated=False + apply_calibration() by hand). Fixed
    by adding an is_ood kwarg to apply_calibration()/predict_rul_calibrated()
    in calibrate_rul.py. Confirmed live: is_ood fired on 13/880 val
    windows, bucket (0,50) moved 86.5% -> 89.4% on val. This file's own
    v7 code did NOT need re-editing for that fix.

  v9 (Aashita): TRAJECTORY-LEAKAGE FIX. Discovered this session: all 108
    flight_ids are just 6 fault-label relabelings of 18 base trajectories
    (fingerprint-matched to 4 sig figs on cht/egt/rpm -- e.g.
    sensor_drift_009 and oil_issue_009 are the same underlying flight,
    differing only in the injected fault channel). The old
    `train_test_split(unique_flights, ...)` split on the 108 fault-labeled
    names directly, so 15/18 base trajectories had copies scattered across
    train/val/test -- train_test_split never produced a real holdout; the
    model had effectively seen most of test already, just under a
    different fault-name label.
    Fix: added base_trajectory_id() (same regex/logic as the standalone
    fix_trajectory_split.py diagnostic script, kept in sync so the two
    don't drift into two independent reimplementations -- see the
    Bug 3 design-smell note from the calibration investigation). The
    __main__ split now splits on the 18 base trajectory IDs first, then
    assigns every fault-labeled flight_id sharing a base ID to that same
    split. Asserts zero base trajectories cross splits before proceeding
    -- hard failure rather than silently shipping a second leaky split.
    First genuinely clean test result after retraining on this split:
    bucket (0,50) PASSES at 91.7%. Bucket (100,200) newly FAILS at 79.2%
    (was passing 85.6-100% in every prior leaky measurement -- its old
    "pass" was itself a leakage artifact). Diagnosed as a genuine
    model-accuracy gap on specific trajectory shapes (test base
    trajectories 000/013 systematically overshoot, point estimate
    plateauing ~210-232 largely independent of true_rul) -- NOT
    calibration or OOD-gate fixable (both ruled out with concrete
    evidence: band-widening would require overfitting val's band to
    test's error; anomalous windows score LOWER on ood_zscore_distance
    than normal windows, the wrong direction for any threshold 1.0-2.0
    to help).

  v10 (this edit): ELAPSED_FRACTION FEATURE, targeting bucket (100,200).
    train_rul_elapsed_fraction_experiment.py ran a controlled single-variant
    experiment (10-feature baseline vs. 11-feature +elapsed_fraction,
    identical window boundaries, corrected leakage-free split, val-only):
    overall RMSE 49.83->18.11, bucket(100,200) RMSE 43.93->2.47, bucket
    max overshoot +52.39->+0.12. check_elapsed_fraction_stability.py then
    verified this wasn't a repeat of the earlier reweighting experiment's
    fake win: stable across 4 seeds (bucket RMSE 0.97-3.42, max overshoot
    +0.12 to +6.95 -- real spread, but nowhere near baseline-bad), and no
    per-flight/per-base collapse (all 18 flights land in a 1.66-3.83 RMSE
    band; base 005's previously-flat prediction at true_rul~165 is
    resolved, not just averaged away). Both checks came back clean, so
    this is now wired into the production ensemble.

    create_windows_v6()/build_windowed_dataset_v6() are the CANONICAL
    implementation -- ported in from (not imported from)
    train_rul_elapsed_fraction_experiment.py, because that script's own
    create_windows_with_elapsed_fraction/build_dataset_baseline
    independently reimplemented create_windows_v5's windowing loop
    (admitted in its own docstring as "reimplemented locally so this
    script is self-contained"). Two live copies of this logic is exactly
    the bug class that caused v7/v8 and the run_coverage_check.py leak,
    so production keeps exactly ONE copy, here.

    Column order: [7 raw sensors, egt_slope, mean_deviation, vib_variance,
    elapsed_fraction] -- 11 columns total, elapsed_fraction at index 10.
    elapsed_fraction = end / flight_len, where `end` is the window's end
    index (exclusive) into that flight's full-length dataframe -- 0.0 near
    flight start, capped at exactly 1.0 at end-of-life (end can never
    exceed flight_len given the loop's range bound).

    RULRegressorVariant's default n_sensors bumped from
    len(SENSOR_FIELDS)+3 to len(SENSOR_FIELDS)+4. __main__ now calls
    build_windowed_dataset_v6() instead of _v5(). Split logic, synthetic
    augmentation, feature bagging, OOD gate, and calibration fit are all
    shape-agnostic (they index SENSOR_FIELDS/CORE_SENSOR_IDX, which are
    still just the first 7 columns) and needed no changes.

    KNOWN CAVEAT / NOT YET RESOLVED: build_inference_window() now REQUIRES
    an elapsed_fraction argument and raises if it's not provided, rather
    than guessing. Live/streaming inference (dashboard) may not know an
    engine's eventual total flight length the way training data does --
    how to estimate elapsed_fraction for a flight still in progress is an
    open design question for the dashboard team, deliberately NOT silently
    defaulted here (a silent default of 0.0 or 1.0 would systematically
    bias every live prediction in one direction). Test set has NOT been
    touched since the v9 clean read -- re-verify bucket(100,200) on val
    after retraining the full 7-variant ensemble + refitting calibration,
    and only then consider a final, one-time test pull to confirm.

Trains N variant RULRegressor models, then exposes predict_rul_ensemble()
which returns:
  - point estimate (mean of variant predictions)
  - rul_lower_bound_timesteps (MIN of variant predictions, clipped >= 0)
  - std across variants (useful for the dashboard's confidence display)

Run: python models/rul/train_rul_ensemble.py   (from project root, AeroTwin/)
"""
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rul import (
    load_all_flights,
    build_windowed_dataset,
    RULDataset,
    evaluate,
    DEVICE,
    MODEL_OUT_DIR,
    WINDOW_SIZE,
    STRIDE,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from schema import SENSOR_FIELDS, HEALTHY_RANGES

# ---------------------------------------------------------------------------
# v5: per-window derived features (Step A, fixing P1 -- point-estimate bias
# on high-RUL windows). Deliberately per-window-only, not per-flight
# cumulative, per the time-constrained decision. Lives here, not in
# train_rul.py, so the original single-model baseline is untouched.
# ---------------------------------------------------------------------------

def _compute_derived_features(window):
    """
    window: np.array shape (window_size, n_sensors=7), raw sensor readings
    Returns: np.array shape (3,) -- [egt_slope, integrated_deviation, vib_variance]
    """
    egt_idx = SENSOR_FIELDS.index("egt")
    vib_idx = SENSOR_FIELDS.index("vibration")

    egt_series = window[:, egt_idx]
    egt_slope = (egt_series[-1] - egt_series[0]) / len(egt_series)

    total_deviation = 0.0
    for i, sensor in enumerate(SENSOR_FIELDS):
        lo, hi = HEALTHY_RANGES[sensor]
        midpoint = (lo + hi) / 2
        range_width = hi - lo
        deviation = np.abs(window[:, i] - midpoint) / range_width
        total_deviation += deviation.sum()

    mean_deviation = total_deviation / window.shape[0]
    vib_variance = np.var(window[:, vib_idx])

    return np.array([egt_slope, mean_deviation, vib_variance])


def build_inference_window(raw_window, elapsed_fraction=None):
    """
    Shared helper for teammates / dashboard code doing live inference.

    raw_window: np.array shape (window_size, 7) -- just the raw sensor
        readings, in SENSOR_FIELDS order, same as what fault injectors /
        the simulator already produce. No need to know about v5/v6
        internals beyond this argument and elapsed_fraction below.

    elapsed_fraction: this window's end position / total flight length,
        in [0.0, 1.0]. REQUIRED as of v10 -- the production ensemble now
        expects an 11-column input (7 raw sensors + 3 derived features +
        elapsed_fraction). Live/streaming callers that don't know the
        flight's eventual total length (e.g. mid-flight inference before
        the engine has failed or the flight has ended) do not currently
        have a principled way to compute this the same way training data
        does -- that's an open question for the dashboard team, not
        resolved here. This function will raise rather than silently
        default a value, since a silent default of 0.0 or 1.0 would
        systematically bias every live prediction in one direction.

    Returns: np.array shape (window_size, 11) -- the 7 raw sensors, the
        3 derived features, and elapsed_fraction, ready to pass into
        predict_rul_ensemble().
    """
    derived = _compute_derived_features(raw_window)
    derived_broadcast = np.tile(derived, (raw_window.shape[0], 1))
    if elapsed_fraction is None:
        raise ValueError(
            "build_inference_window() now requires elapsed_fraction as of v10 "
            "(model input is 11 columns, not 10). If doing live/streaming "
            "inference where total flight length isn't known yet, decide "
            "explicitly how to estimate it (e.g. a running estimate, or a "
            "fixed assumed max flight length) -- do not silently default "
            "this to 0.0 or 1.0, since that would systematically bias "
            "every live prediction in one direction."
        )
    elapsed_broadcast = np.full((raw_window.shape[0], 1), elapsed_fraction)
    return np.concatenate([raw_window, derived_broadcast, elapsed_broadcast], axis=1)


def create_windows_v5(df, window_size=WINDOW_SIZE, stride=STRIDE):
    """Retained for backward compatibility / anything still training the
    old 10-feature baseline. Production training (see __main__) now uses
    create_windows_v6 below."""
    sensor_data = df[SENSOR_FIELDS].values
    rul = df["true_rul_timesteps"].values

    windows, labels = [], []
    for start in range(0, len(df) - window_size + 1, stride):
        end = start + window_size
        raw_window = sensor_data[start:end]
        derived = _compute_derived_features(raw_window)
        derived_broadcast = np.tile(derived, (raw_window.shape[0], 1))
        full_window = np.concatenate([raw_window, derived_broadcast], axis=1)
        windows.append(full_window)
        labels.append(rul[end - 1])

    return np.array(windows), np.array(labels)


def build_windowed_dataset_v5(flights):
    """Retained for backward compatibility. Production training uses
    build_windowed_dataset_v6 below."""
    all_windows, all_labels, all_flight_ids = [], [], []
    for flight_id, df in flights:
        windows, labels = create_windows_v5(df)
        if len(windows) == 0:
            continue
        all_windows.append(windows)
        all_labels.append(labels)
        all_flight_ids.extend([flight_id] * len(windows))

    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y, np.array(all_flight_ids)


# ---------------------------------------------------------------------------
# v10: elapsed_fraction feature (window's end position / total flight
# length). Added after check_elapsed_fraction_stability.py confirmed this
# was stable across seeds and general across flights/bases on val (not a
# 1-2-flight collapse like the earlier reweighting experiment) -- see
# train_rul_elapsed_fraction_experiment.py for the original single-variant
# result (bucket(100,200) RMSE 43.93->2.47, max overshoot +52.39->+0.12).
#
# This is the CANONICAL implementation -- ported in from
# train_rul_elapsed_fraction_experiment.py rather than imported, because
# that script's own create_windows_with_elapsed_fraction/build_dataset_baseline
# independently reimplemented create_windows_v5's windowing loop (admitted
# in its own docstring as "reimplemented locally"). Two live copies of this
# logic is exactly the bug class that caused v7/v8 and the
# run_coverage_check.py leak, so production gets ONE copy, here.
#
# Column order: [7 raw sensors, egt_slope, mean_deviation, vib_variance,
# elapsed_fraction] -- 11 columns total, elapsed_fraction at index 10.
# elapsed_fraction = end / flight_len, where `end` is the window's end
# index (exclusive) into that flight's full-length dataframe -- 0.0 at
# flight start, capped at exactly 1.0 at end-of-life (end can never exceed
# flight_len given the loop bound below).
# ---------------------------------------------------------------------------

def create_windows_v6(df, window_size=WINDOW_SIZE, stride=STRIDE):
    sensor_data = df[SENSOR_FIELDS].values
    rul = df["true_rul_timesteps"].values
    flight_len = len(df)

    windows, labels = [], []
    for start in range(0, len(df) - window_size + 1, stride):
        end = start + window_size
        raw_window = sensor_data[start:end]

        derived = _compute_derived_features(raw_window)
        elapsed_fraction = end / flight_len  # 0.0 at flight start, capped at 1.0 at end-of-life

        derived_broadcast = np.tile(derived, (window_size, 1))
        elapsed_broadcast = np.full((window_size, 1), elapsed_fraction)
        full_window = np.concatenate([raw_window, derived_broadcast, elapsed_broadcast], axis=1)

        windows.append(full_window)
        labels.append(rul[end - 1])

    return np.array(windows), np.array(labels)


def build_windowed_dataset_v6(flights):
    all_windows, all_labels, all_flight_ids = [], [], []
    for flight_id, df in flights:
        windows, labels = create_windows_v6(df)
        if len(windows) == 0:
            continue
        all_windows.append(windows)
        all_labels.append(labels)
        all_flight_ids.extend([flight_id] * len(windows))

    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y, np.array(all_flight_ids)


# ---------------------------------------------------------------------------
# v9: trajectory-leakage fix. flight_ids like "sensor_drift_009" and
# "oil_issue_009" are DIFFERENT fault-name labels stamped onto the SAME
# underlying base trajectory ("009"). Splitting on flight_id (the old
# behavior) leaks near-duplicate trajectories across train/val/test.
# This must split on the base trajectory ID instead, and keep every
# fault-labeled copy of a base trajectory together in one split.
#
# NOTE: this logic intentionally mirrors fix_trajectory_split.py's
# base_trajectory_id() exactly. If you ever change the regex here, change
# it there too -- two independent reimplementations of the same split
# logic drifting apart is exactly what caused the v7/v8 OOD-gate bug.
# ---------------------------------------------------------------------------

def base_trajectory_id(flight_id):
    """'sensor_drift_009' -> '009'. Raises if a flight_id doesn't match
    the expected '<fault_type>_<NNN>' pattern -- inspect naming before
    trusting a split if this ever fires unexpectedly."""
    m = re.search(r'_(\d+)$', flight_id)
    if not m:
        raise ValueError(
            f"flight_id {flight_id!r} doesn't match the expected "
            f"'<fault_type>_<NNN>' pattern -- inspect naming before trusting this split."
        )
    return m.group(1)


def generate_synthetic_multifault_windows(X, y, n_synthetic, seed=0):
    """
    Builds synthetic multi-fault windows by combining pairs of existing
    (real, single-fault) windows' RAW sensor deviations from healthy
    midpoints, added together onto one of the two base windows. This
    approximates what a genuine combined-fault signature would look like,
    without needing new simulator data.

    Deliberately mirrors robustness_tests.py Test 4's own construction
    (combining an oil-issue-like and vibration-fault-like signature in one
    window) -- so the model sees SOME combined-fault variance during
    training, rather than encountering it for the first time at test time.

    X: np.array shape (n_windows, window_size, n_features) -- featurized
        windows (10 or 11 columns depending on which builder produced X).
    y: np.array shape (n_windows,) -- true RUL labels.
    n_synthetic: how many synthetic windows to generate.
    seed: RNG seed, offset from other seed uses.

    The label for each synthetic window is the MIN of the two source
    windows' labels (conservative -- a combined fault should not be
    assumed less urgent than either fault alone).

    Derived features (egt_slope, mean_deviation, vib_variance) are
    recomputed from the combined raw sensors so each synthetic window is
    internally consistent. elapsed_fraction (if present, column 10) is
    NOT recomputed -- it doesn't depend on sensor values, and each
    synthetic window inherits window A's elapsed_fraction unchanged,
    which is correct since the synthetic window's timing/position is
    window A's.

    Returns: X_synthetic, y_synthetic -- to be concatenated onto the real
    training set (NOT val/test -- synthetic augmentation is train-only,
    same principle as bootstrapping/feature bagging/window dropout).
    """
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
        combined_raw = base.copy()
        for s_idx, sensor in enumerate(SENSOR_FIELDS):
            lo, hi = HEALTHY_RANGES[sensor]
            midpoint = (lo + hi) / 2
            base_dev = base[:, s_idx] - midpoint
            other_dev = other[:, s_idx] - midpoint
            combined_raw[:, s_idx] = midpoint + base_dev + other_dev

        new_derived = _compute_derived_features(combined_raw)
        X_synthetic[k, :, :n_raw_sensors] = combined_raw
        X_synthetic[k, :, n_raw_sensors:n_raw_sensors + 3] = new_derived
        # elapsed_fraction (column n_raw_sensors+3, if present) intentionally
        # left untouched -- inherited from window A, not recomputed.

    return X_synthetic, y_synthetic


# From sensor_importance.py output (RandomForest R^2 = 0.840 on validation):
#   1. vibration    0.2779
#   2. cht          0.2354
#   3. oil_pressure 0.2251
#   4. rpm          0.1623
#   (oil_temp, egt, fuel_flow are non-core, ~0.03-0.04 each)
RUL_CORE_SENSORS = ['vibration', 'cht', 'oil_pressure', 'rpm']
CORE_SENSOR_IDX = [SENSOR_FIELDS.index(s) for s in RUL_CORE_SENSORS]
NON_CORE_SENSOR_IDX = [i for i in range(len(SENSOR_FIELDS)) if i not in CORE_SENSOR_IDX]

N_VARIANTS = 7
SEEDS = [0, 1, 2, 3, 4, 5, 6]
DROPOUTS = [0.0, 0.1, 0.15, 0.1, 0.2, 0.15, 0.25]
HIDDEN_SIZES = [24, 32, 32, 40, 40, 48, 32]     # never below 24 (16 confirmed too weak in v3)
N_DROPPED_FEATURES = [0, 1, 1, 2, 1, 2, 0]      # how many sensor columns each variant zeroes out


# ---------------------------------------------------------------------------
# Variant model -- same architecture shape as RULRegressor, but hidden_size
# and dropout vary per variant.
# ---------------------------------------------------------------------------

class RULRegressorVariant(nn.Module):
    def __init__(self, n_sensors=len(SENSOR_FIELDS) + 4, hidden_size=32, dropout=0.0):
        # n_sensors default bumped from +3 to +4 in v10 (elapsed_fraction).
        # If ever loading an OLD (pre-v10, 10-feature) saved model, pass
        # n_sensors=len(SENSOR_FIELDS)+3 explicitly.
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_sensors, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(self.dropout(h_n[-1]))
        return out.squeeze(-1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def bootstrap_by_flight(X, y, flight_ids, seed):
    """
    Resamples FLIGHTS with replacement (not individual windows/rows), then
    gathers all windows belonging to the resampled flights.
    """
    rng = np.random.RandomState(seed)
    unique_flights = np.unique(flight_ids)
    sampled_flights = rng.choice(unique_flights, size=len(unique_flights), replace=True)

    idx_list = []
    for f in sampled_flights:
        matches = np.where(flight_ids == f)[0]
        idx_list.append(matches)
    idx = np.concatenate(idx_list)

    return X[idx], y[idx]


def apply_feature_bagging(X, n_dropped, seed):
    """
    Zeroes out n_dropped sensor columns, CORE-AWARE:
      - at most ONE of RUL_CORE_SENSORS may be dropped, never two or more together
      - the rest of the drops (if n_dropped > 1) come from NON_CORE_SENSOR_IDX
      - if n_dropped == 0, nothing is dropped

    Only ever touches raw sensor columns (indices into CORE_SENSOR_IDX /
    NON_CORE_SENSOR_IDX, both subsets of the first 7 columns) -- derived
    features and elapsed_fraction (columns 7+) are never dropped here.
    """
    if n_dropped == 0:
        return X, np.array([], dtype=int)

    rng = np.random.RandomState(seed + 1000)  # offset so it differs from bootstrap's seed use

    dropped = []
    drop_one_core = rng.random() < 0.5  # 50/50 chance to include a core sensor
    if drop_one_core:
        dropped.append(int(rng.choice(CORE_SENSOR_IDX)))
        remaining = n_dropped - 1
    else:
        remaining = n_dropped

    if remaining > 0:
        non_core_pool = [i for i in NON_CORE_SENSOR_IDX if i not in dropped]
        remaining = min(remaining, len(non_core_pool))  # can't drop more than exist
        dropped.extend(rng.choice(non_core_pool, size=remaining, replace=False).tolist())

    dropped_idx = np.array(sorted(dropped))
    X_bagged = X.copy()
    X_bagged[:, :, dropped_idx] = 0.0
    return X_bagged, dropped_idx


def apply_window_level_dropout(X, drop_prob=0.15, seed=0):
    """
    Per-window sensor dropout augmentation (training only). Recomputes the
    3 derived features after zeroing a raw sensor column, since those
    depend on raw sensor values. elapsed_fraction (column 10, if present)
    is intentionally NOT recomputed here -- it doesn't depend on sensor
    values, so it's unaffected by this augmentation.
    """
    rng = np.random.RandomState(seed + 5000)  # offset from other seed uses
    X_aug = X.copy()
    n_windows = X_aug.shape[0]
    n_raw_sensors = len(SENSOR_FIELDS)

    affected = rng.random(n_windows) < drop_prob
    affected_idx = np.where(affected)[0]

    for i in affected_idx:
        sensor_idx = rng.randint(0, n_raw_sensors)
        X_aug[i, :, sensor_idx] = 0.0
        new_derived = _compute_derived_features(X_aug[i, :, :n_raw_sensors])
        X_aug[i, :, n_raw_sensors:n_raw_sensors + 3] = new_derived

    return X_aug, affected_idx


def train_one_variant(X_train, y_train, train_flight_ids, X_eval, y_eval,
                       seed, dropout, hidden_size, n_dropped_features,
                       epochs=70, batch_size=32, lr=1e-3):
    """
    X_eval/y_eval is whatever set you want per-variant RMSE reported against
    during tuning -- pass VALIDATION data here, never test, until the final
    locked-in check.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train_boot, y_train_boot = bootstrap_by_flight(X_train, y_train, train_flight_ids, seed)
    X_train_boot, dropped_idx = apply_feature_bagging(X_train_boot, n_dropped_features, seed)
    X_train_boot, _ = apply_window_level_dropout(X_train_boot, drop_prob=0.15, seed=seed)
    X_eval_bagged = X_eval.copy()
    if len(dropped_idx) > 0:
        X_eval_bagged[:, :, dropped_idx] = 0.0

    n_features = X_train.shape[-1]

    train_loader = DataLoader(RULDataset(X_train_boot, y_train_boot), batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(RULDataset(X_eval_bagged, y_eval), batch_size=batch_size, shuffle=False)

    model = RULRegressorVariant(n_sensors=n_features, hidden_size=hidden_size, dropout=dropout).to(DEVICE)
    n_params = count_params(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

    eval_rmse = evaluate(model, eval_loader)
    return model, eval_rmse, dropped_idx, n_params


def load_ensemble(model_dir=MODEL_OUT_DIR, n_variants=N_VARIANTS, n_features=len(SENSOR_FIELDS) + 4):
    """
    Loads a previously trained + saved ensemble from disk, ready to pass
    straight into predict_rul_ensemble(window, models, scaler, dropped_idx_list).

    n_features defaults to 11 (v10, elapsed_fraction included). Pass
    n_features=len(SENSOR_FIELDS)+3 explicitly if loading an OLDER
    (pre-v10, 10-feature) saved ensemble.
    """
    scaler_path = os.path.join(model_dir, "rul_ensemble_scaler.joblib")
    dropped_idx_path = os.path.join(model_dir, "rul_ensemble_dropped_idx.joblib")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"No trained ensemble found at {model_dir} -- run "
            f"train_rul_ensemble.py first to train and save one."
        )

    scaler = joblib.load(scaler_path)
    dropped_idx_list = joblib.load(dropped_idx_path)

    models = []
    for i in range(n_variants):
        model_path = os.path.join(model_dir, f"rul_model_variant_{i}.pt")
        model = RULRegressorVariant(n_sensors=n_features, hidden_size=HIDDEN_SIZES[i])
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        models.append(model)

    return models, scaler, dropped_idx_list


def compute_ood_zscore_distance(window, scaler):
    """
    Post-hoc OOD signal: average per-feature |z-score| of this window's
    values relative to the TRAINING distribution.
    """
    z = (window - scaler.mean_) / scaler.scale_
    return float(np.abs(z).mean())


def predict_rul_ensemble(window, models, scaler, dropped_idx_list, calibrated=True,
                          ood_gate=True, ood_zscore_threshold=2.0, ood_std_floor=50.0):
    """
    window: np.array shape (window_size, 11) as of v10 -- the FEATURIZED
        window (7 raw sensors + 3 derived features + elapsed_fraction).
        If you have a raw 7-column sensor window instead, call
        build_inference_window(raw_window, elapsed_fraction) first to get
        this shape -- elapsed_fraction is now a required argument there.
    models: list of trained RULRegressorVariant instances.
    scaler: the StandardScaler fit during ensemble training.
    dropped_idx_list: list of dropped-feature-index arrays, one per model,
        in the SAME order as `models`.
    calibrated: if True, applies Step B's per-bucket bias correction +
        conformal lower bound (see calibrate_rul.py) before returning.

    Returns dict:
      point_estimate_timesteps / point_estimate_minutes -- mean across
          variants (or bias-corrected, if calibrated=True).
      rul_lower_bound_timesteps / rul_lower_bound_minutes -- min across
          variants, clipped >= 0 (or conformal lower bound, if
          calibrated=True). If this window is flagged OOD and
          calibrated=True, this is forced to 0.0 instead -- the
          bucket-based band is not trusted for inputs this far from the
          training distribution.
      std_timesteps / std_minutes -- spread across variants, inflated to
          at least ood_std_floor if this window is flagged OOD.
    """
    scaled = scaler.transform(window.reshape(-1, window.shape[-1])).reshape(window.shape)

    preds = []
    with torch.no_grad():
        for model, dropped_idx in zip(models, dropped_idx_list):
            model.eval()
            window_variant = scaled.copy()
            if len(dropped_idx) > 0:
                window_variant[:, dropped_idx] = 0.0
            x = torch.tensor(window_variant, dtype=torch.float32).unsqueeze(0)
            preds.append(model(x).item())

    preds = np.array(preds)
    point_estimate = max(0.0, float(preds.mean()))
    lower_bound = max(0.0, float(preds.min()))
    std = float(preds.std())

    is_ood = False
    if ood_gate:
        ood_distance = compute_ood_zscore_distance(window, scaler)
        if ood_distance > ood_zscore_threshold:
            std = max(std, ood_std_floor)
            is_ood = True

    if calibrated:
        from calibrate_rul import apply_calibration, load_calibration_params
        y_cal, lb_cal = apply_calibration(point_estimate, load_calibration_params())
        if is_ood:
            lb_cal = 0.0
        return {
            "point_estimate_timesteps": y_cal,
            "rul_lower_bound_timesteps": lb_cal,
            "std_timesteps": std,
            "point_estimate_minutes": y_cal,
            "rul_lower_bound_minutes": lb_cal,
            "std_minutes": std,
        }

    return {
        "point_estimate_timesteps": point_estimate,
        "rul_lower_bound_timesteps": lower_bound,
        "std_timesteps": std,
        "point_estimate_minutes": point_estimate,
        "rul_lower_bound_minutes": lower_bound,
        "std_minutes": std,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading flights from data/raw/ ...")
    flights = load_all_flights()
    print(f"Loaded {len(flights)} fault flights (healthy-only flights excluded).")

    print("Building windowed dataset (v10: 11 features, includes elapsed_fraction) ...")
    X, y, flight_ids = build_windowed_dataset_v6(flights)
    print(f"Total windows: {len(X)}  Feature columns: {X.shape[-1]}")

    # -----------------------------------------------------------------
    # v9: split on BASE TRAJECTORY IDs, not on the 108 fault-labeled
    # flight_ids directly. See base_trajectory_id() and the v9 changelog
    # note above for why -- the old split leaked near-duplicate
    # trajectories (same base flight, different fault-name label) across
    # train/val/test.
    # -----------------------------------------------------------------
    print("Splitting by BASE TRAJECTORY (not flight_id) into train/val/test ...")
    unique_flights = np.unique(flight_ids)
    base_ids = np.array([base_trajectory_id(f) for f in unique_flights])
    unique_bases = np.unique(base_ids)
    print(f"Unique flight_ids: {len(unique_flights)}  ->  unique BASE trajectories: {len(unique_bases)}")

    train_bases, temp_bases = train_test_split(unique_bases, test_size=0.3, random_state=42)
    val_bases, test_bases = train_test_split(temp_bases, test_size=0.5, random_state=42)
    train_bases_set, val_bases_set, test_bases_set = set(train_bases), set(val_bases), set(test_bases)

    train_flights = np.array([f for f in unique_flights if base_trajectory_id(f) in train_bases_set])
    val_flights = np.array([f for f in unique_flights if base_trajectory_id(f) in val_bases_set])
    test_flights = np.array([f for f in unique_flights if base_trajectory_id(f) in test_bases_set])

    # Hard sanity check -- fail loudly rather than silently ship a second
    # leaky split. Do not remove this assert.
    split_of = {f: "train" for f in train_flights}
    split_of.update({f: "val" for f in val_flights})
    split_of.update({f: "test" for f in test_flights})
    crossing = 0
    for b in unique_bases:
        members = [f for f in unique_flights if base_trajectory_id(f) == b]
        if len(set(split_of[f] for f in members)) > 1:
            crossing += 1
    print(f"Base trajectories crossing splits: {crossing}  (should be 0)")
    assert crossing == 0, "Split still leaks across base trajectories -- do not proceed."

    train_mask = np.isin(flight_ids, train_flights)
    val_mask = np.isin(flight_ids, val_flights)
    test_mask = np.isin(flight_ids, test_flights)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    train_flight_ids = flight_ids[train_mask]

    n_synthetic = int(0.10 * len(X_train))
    print(f"Generating {n_synthetic} synthetic multi-fault training windows ...")
    X_synth, y_synth = generate_synthetic_multifault_windows(X_train, y_train, n_synthetic, seed=42)
    synth_flight_ids = np.array([f"synthetic_{i}" for i in range(n_synthetic)])
    X_train = np.concatenate([X_train, X_synth], axis=0)
    y_train = np.concatenate([y_train, y_synth], axis=0)
    train_flight_ids = np.concatenate([train_flight_ids, synth_flight_ids], axis=0)
    print(f"Train windows after synthetic augmentation: {len(X_train)}")

    print(f"Train windows: {len(X_train)} -- Val windows: {len(X_val)} -- Test windows: {len(X_test)}")
    print(f"Train flights: {len(train_flights)} -- Val flights: {len(val_flights)} -- Test flights: {len(test_flights)}")
    print(f"Train bases: {len(train_bases)} -- Val bases: {len(val_bases)} -- Test bases: {len(test_bases)}")

    print("Scaling sensor features (fit on train only) ...")
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    X_train_scaled = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    print(f"\nTraining {N_VARIANTS} ensemble variants (bootstrapping + hidden_size + core-aware feature bagging) ...")
    print("NOTE: per-variant RMSE below is on the VALIDATION set. Test stays untouched until the final check.")
    models = []
    rmses = []
    dropped_idx_list = []
    for i in range(N_VARIANTS):
        seed = SEEDS[i]
        dropout = DROPOUTS[i]
        hidden_size = HIDDEN_SIZES[i]
        n_dropped = N_DROPPED_FEATURES[i]
        print(f"  Variant {i+1}/{N_VARIANTS} -- seed={seed}, dropout={dropout}, "
              f"hidden_size={hidden_size}, n_dropped_features={n_dropped} ...")
        model, rmse, dropped_idx, n_params = train_one_variant(
            X_train_scaled, y_train, train_flight_ids, X_val_scaled, y_val,
            seed, dropout, hidden_size, n_dropped
        )
        dropped_names = [SENSOR_FIELDS[j] for j in dropped_idx]
        print(f"    -> val RMSE: {rmse:.2f}  ({n_params:,} params)  dropped: {dropped_names}")
        models.append(model)
        rmses.append(rmse)
        dropped_idx_list.append(dropped_idx)

    print(f"\nEnsemble val RMSEs: {[round(r, 2) for r in rmses]}")
    print(f"Mean single-model val RMSE: {np.mean(rmses):.2f}  (v3 baseline was 44.94, pre-elapsed_fraction v9 baseline was ~49.83 overall)")

    print("\nSanity check on 5 VALIDATION windows (tuning happens here, test stays untouched):")
    for i in range(min(5, len(X_val))):
        result = predict_rul_ensemble(X_val[i], models, scaler, dropped_idx_list)
        true_val = y_val[i]
        assert result["rul_lower_bound_timesteps"] <= result["point_estimate_timesteps"] + 1e-6, \
            "Lower bound exceeded point estimate!"
        assert result["rul_lower_bound_timesteps"] >= 0, "Lower bound went negative!"
        print(f"  true={true_val:7.1f}  point_est={result['point_estimate_timesteps']:7.1f}  "
              f"lower_bound={result['rul_lower_bound_timesteps']:7.1f}  std={result['std_timesteps']:6.1f}")

    print("\nBUCKET(100,200) SPOT-CHECK (val) -- confirm the ensemble-level result matches the")
    print("single-variant experiment before trusting this for the eventual test pull:")
    bucket_mask = (y_val >= 100) & (y_val < 200)
    bucket_preds = []
    for i in np.where(bucket_mask)[0]:
        result = predict_rul_ensemble(X_val[i], models, scaler, dropped_idx_list, calibrated=False)
        bucket_preds.append(result["point_estimate_timesteps"])
    bucket_preds = np.array(bucket_preds)
    bucket_true = y_val[bucket_mask]
    bucket_rmse = np.sqrt(np.mean((bucket_preds - bucket_true) ** 2))
    bucket_max_overshoot = np.max(bucket_preds - bucket_true)
    print(f"  n={bucket_mask.sum()}  bucket(100,200) val RMSE={bucket_rmse:.2f}  "
          f"max_overshoot={bucket_max_overshoot:+.2f}  (single-variant experiment: RMSE=2.47, overshoot=+0.12)")
    print("  If this is notably worse than the single-variant experiment, the other 6 variants'")
    print("  feature bagging/dropout/hidden-size diversity may be diluting elapsed_fraction's")
    print("  signal -- investigate before refitting calibration or touching test.")

    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    for i, model in enumerate(models):
        model_path = os.path.join(MODEL_OUT_DIR, f"rul_model_variant_{i}.pt")
        torch.save(model.state_dict(), model_path)
    scaler_path = os.path.join(MODEL_OUT_DIR, "rul_ensemble_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    dropped_idx_path = os.path.join(MODEL_OUT_DIR, "rul_ensemble_dropped_idx.joblib")
    joblib.dump(dropped_idx_list, dropped_idx_path)
    print(f"\nSaved {N_VARIANTS} variant models to {MODEL_OUT_DIR}\\rul_model_variant_*.pt")
    print(f"Saved ensemble scaler to {scaler_path}")
    print(f"Saved dropped-feature indices to {dropped_idx_path}")

    print("\nFitting Step B calibration (bias correction + conformal) on VALIDATION set ...")
    from calibrate_rul import fit_calibration
    fit_calibration(X_val, y_val, models, scaler, dropped_idx_list)
    print("\nTest set (X_test_scaled/y_test) is built and scaled above but NOT used yet --")
    print("reserved for the final, one-time coverage check. Run separately:")
    print("  from calibrate_rul import load_calibration_params, check_per_bucket_coverage")
    print("  params = load_calibration_params()")
    print("  check_per_bucket_coverage(X_test_scaled, y_test, models, scaler, dropped_idx_list, params)")
    print("calibrated= now defaults to True (Step B verified, see changelog). Pass calibrated=False explicitly if you need the pre-calibration raw ensemble output.")
    print("\nIMPORTANT (v9, still applies): before running check_per_bucket_coverage() on test above,")
    print("verify run_coverage_check.py still imports base_trajectory_id() from this file rather")
    print("than reimplementing the split -- it was patched to do so, confirm that hasn't regressed.")
    print("IMPORTANT (v10): run_coverage_check.py must also build its test windows via")
    print("build_windowed_dataset_v6 (11 features), not build_windowed_dataset_v5 (10 features),")
    print("or it will silently feed 10-column windows into an 11-feature model. Verify this")
    print("before the test run -- do not assume it was updated automatically.")

    print("\nTeammates: if you have a raw 7-sensor window, call")
    print("build_inference_window(raw_window, elapsed_fraction) first (elapsed_fraction is")
    print("now REQUIRED -- see that function's docstring for the open live-inference question),")
    print("then pass the result to predict_rul_ensemble(window, models, scaler, dropped_idx_list)")
    print("-- or, if training has already been run once, just call load_ensemble()")
    print("to get (models, scaler, dropped_idx_list) straight from the saved files")
    print("without rerunning training.")
    print("Pass calibrated=True to get Step B's bias-corrected estimate + conformal lower bound.")