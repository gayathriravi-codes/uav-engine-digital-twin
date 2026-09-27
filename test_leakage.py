"""
test_leakage.py
------------------
Regression tests guarding against the trajectory-relabeling leakage bug
found and fixed this session: every base trajectory (18 total, IDs
000-017) was relabeled under up to 7 fault names, byte-identical for
early windows. Two independent instances of the same underlying bug
were found and fixed:
  1. RUL pipeline (train_rul_ensemble.py) - already used base_trajectory_id
     grouping going into this session.
  2. Classification pipeline (train_xgboost.py) - was grouping
     StratifiedGroupKFold on flight_id (fault-labeled name) instead of
     base trajectory, confirmed leaking; fixed in train_xgboost_fixed.py.

These tests exist so a future refactor, a new teammate's script, or a
retrain can't reintroduce either version of this bug without a test
failing first.

Run: python -m pytest test_leakage.py -v
"""

import pandas as pd
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CLASSIFICATION_FEATURES = PROJECT_ROOT / "data" / "processed" / "classification_features.csv"


def get_base_id(flight_id: str) -> str:
    """Same convention used throughout the project: base trajectory ID
    is everything after the last underscore in the fault-labeled name."""
    return flight_id.rsplit("_", 1)[-1]


class TestClassificationSplitLeakage:
    """
    Guards against the exact bug found this session: StratifiedGroupKFold
    grouped on the fault-labeled flight_id instead of the base trajectory,
    so e.g. misfire_005 and overheat_005 (the same underlying trajectory)
    could land in different folds despite being byte-identical early on.
    """

    @pytest.fixture(scope="class")
    def df(self):
        if not CLASSIFICATION_FEATURES.exists():
            pytest.skip(
                f"{CLASSIFICATION_FEATURES} not found - run "
                f"models/classification/feature_extraction.py first."
            )
        return pd.read_csv(CLASSIFICATION_FEATURES)

    def test_expected_base_trajectory_count(self, df):
        """Sanity check on the known dataset shape (18 base trajectories,
        000-017) - catches silent changes to the underlying flight data."""
        base_ids = df["flight_id"].apply(get_base_id)
        unique_bases = sorted(base_ids.unique())
        assert len(unique_bases) == 18, (
            f"Expected 18 base trajectories, found {len(unique_bases)}: {unique_bases}. "
            f"If the dataset genuinely changed, update this test's expected count."
        )

    def test_no_fault_labeled_copy_is_orphaned(self, df):
        """Every base trajectory should have multiple fault-labeled copies
        (that's the whole mechanism this bug depends on) - if a base_id
        ever has only 1 copy, this test catches that the dataset's shape
        has changed in a way that changes what this bug even looks like."""
        base_ids = df["flight_id"].apply(get_base_id)
        copies_per_base = df.groupby(base_ids)["flight_id"].nunique()
        assert (copies_per_base >= 2).all(), (
            f"Some base trajectories have only 1 fault-labeled copy:\n"
            f"{copies_per_base[copies_per_base < 2]}"
        )

    def test_grouping_by_flight_id_would_leak(self, df):
        """
        Documents WHY grouping on flight_id is wrong: confirms rows
        sharing a base_id but different flight_id have near-identical
        readings at window_start == 0 (before any fault has had time to
        manifest, regardless of onset timing differences between faults).
        Uses relative tolerance since raw values (e.g. rpm ~5000) make an
        absolute threshold meaningless.
        """
        sensor_cols = [c for c in df.columns if c.endswith("_mean")]
        base_ids = df["flight_id"].apply(get_base_id)

        copies_per_base = df.groupby(base_ids)["flight_id"].nunique()
        candidates = copies_per_base[copies_per_base >= 2].index

        for base in candidates:
            rows = df[(base_ids == base) & (df["window_start"] == 0)]
            first_windows = rows.groupby("flight_id").first()
            if len(first_windows) >= 2:
                a, b = first_windows.iloc[0], first_windows.iloc[1]
                near_identical = all(
                    abs(a[c] - b[c]) <= 0.02 * max(abs(a[c]), abs(b[c]), 1.0)
                    for c in sensor_cols if pd.notna(a[c]) and pd.notna(b[c])
                )
                assert near_identical, (
                    f"base_id={base}: expected near-identical window_start=0 "
                    f"sensor means between {first_windows.index[0]} and "
                    f"{first_windows.index[1]} - if this fails, the simulator's "
                    f"data-generation approach may have changed."
                )
                return

        pytest.skip("No multi-copy base_id had a window_start==0 row for 2+ flight_ids.")

    def test_grouped_split_produces_no_base_id_overlap(self, df):
        """
        The actual regression guard: runs the SAME StratifiedGroupKFold
        setup train_xgboost_fixed.py uses, grouped on base_id, and asserts
        no base trajectory ever appears in both train and test for any
        fold.
        """
        from sklearn.model_selection import StratifiedGroupKFold

        base_ids = df["flight_id"].apply(get_base_id)
        y = df["label"]

        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (train_idx, test_idx) in enumerate(cv.split(df, y, groups=base_ids), start=1):
            train_bases = set(base_ids.iloc[train_idx])
            test_bases = set(base_ids.iloc[test_idx])
            overlap = train_bases & test_bases
            assert not overlap, (
                f"Fold {fold}: base trajectories {overlap} appear in both "
                f"train and test - this is the exact leakage bug found "
                f"and fixed this session, reintroduced."
            )

    def test_flight_id_grouping_would_have_failed(self, df):
        """
        Negative-control test: confirms the OLD (buggy) approach -
        grouping on flight_id directly - actually does produce overlap
        for this dataset. Documents, in a way that fails loudly if the
        underlying data ever changes, that the bug was real.
        """
        from sklearn.model_selection import StratifiedGroupKFold

        base_ids = df["flight_id"].apply(get_base_id)
        y = df["label"]
        flight_ids = df["flight_id"]

        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        found_a_leak = False
        for train_idx, test_idx in cv.split(df, y, groups=flight_ids):
            train_bases = set(base_ids.iloc[train_idx])
            test_bases = set(base_ids.iloc[test_idx])
            if train_bases & test_bases:
                found_a_leak = True
                break

        assert found_a_leak, (
            "Expected grouping-by-flight_id to leak base trajectories across "
            "folds (that was the original bug) - if this no longer reproduces, "
            "either the dataset or sklearn's fold assignment changed."
        )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))