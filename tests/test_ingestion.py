"""
tests/test_ingestion.py -- run from the repo root:  python tests/test_ingestion.py
(also works with pytest). Uses its own synthetic data, so no simulator needed.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.validator import SENSORS, validate  # noqa: E402


def clean(n=300, seed=0):
    r = np.random.default_rng(seed)
    d = {"timestamp": np.arange(n)}
    base = {"rpm": 5000, "egt": 700, "cht": 180, "oil_pressure": 60,
            "oil_temp": 90, "vibration": 5, "fuel_flow": 20}
    for s in SENSORS:
        d[s] = base[s] + r.normal(0, base[s] * 0.01, n)
    return pd.DataFrame(d)


def codes(res):
    return {i.code for i in res.issues}


def test_clean_ok():
    res = validate(clean(), {"time_column": "timestamp"})
    assert res.status == "OK", res.report()
    assert len(res.segments) == 1 and len(res.segments[0]) == 300
    assert list(res.segments[0].columns) == ["timestamp"] + SENSORS


def test_rename_and_units():
    df = clean()
    df["CHT_F"] = df["cht"] * 9 / 5 + 32
    df = df.drop(columns=["cht"])
    cfg = {"time_column": "timestamp",
           "sensors": {"cht": {"column": "CHT_F", "scale": 5 / 9, "offset": -160 / 9}}}
    res = validate(df, cfg)
    assert res.status == "OK", res.report()
    assert abs(res.segments[0]["cht"].mean() - 180) < 2


def test_missing_sensor_rejects():
    res = validate(clean().drop(columns=["oil_temp"]), {"time_column": "timestamp"})
    assert res.status == "REJECT" and "MISSING_SENSOR" in codes(res)
    assert not res.segments


def test_short_gap_filled_long_gap_splits():
    df = clean(400)
    df.loc[50:51, "egt"] = np.nan          # 2 ticks: filled
    df.loc[200:215, "rpm"] = np.nan        # 16 ticks: splits the flight
    res = validate(df, {"time_column": "timestamp"})
    assert "GAP_FILLED" in codes(res) and "LONG_GAP" in codes(res)
    assert len(res.segments) == 2
    assert res.status == "WARN"


def test_order_and_duplicates():
    df = clean()
    df = pd.concat([df, df.iloc[[10, 11]]]).sample(frac=1, random_state=1)
    res = validate(df, {"time_column": "timestamp"})
    assert {"OUT_OF_ORDER", "DUPLICATE_TIMESTAMPS"} <= codes(res)
    assert len(res.segments[0]) == 300


def test_out_of_range_blanked_not_clipped():
    df = clean()
    df.loc[100, "rpm"] = -500
    res = validate(df, {"time_column": "timestamp"})
    assert "OUT_OF_RANGE" in codes(res)
    assert (res.segments[0]["rpm"] > 0).all()       # interpolated over, never -500 or 0


def test_slow_sampling_flagged():
    df = clean(200)
    df["timestamp"] = df["timestamp"] * 2.0
    res = validate(df, {"time_column": "timestamp", "max_gap_ticks": 1})
    assert "RESAMPLED" in codes(res)


def test_mostly_bad_rejects():
    df = clean()
    df.loc[20:280, "vibration"] = np.nan
    res = validate(df, {"time_column": "timestamp"})
    assert res.status == "REJECT" and "LOW_USABLE" in codes(res)


def test_flat_sensor_warns():
    df = clean()
    df["oil_pressure"] = 60.0
    res = validate(df, {"time_column": "timestamp"})
    assert "FLAT_SENSOR" in codes(res)


def test_long_gap_not_partially_filled():
    # regression: a gap longer than max_gap must get NO fill, not a filled edge
    df = clean(300)
    df.loc[100:110, "egt"] = np.nan        # 11 ticks
    res = validate(df, {"time_column": "timestamp"})
    assert "GAP_FILLED" not in codes(res), res.report()
    assert [s["ticks"] for s in res.segment_info] == [100, 189], res.segment_info


def test_missing_not_labeled_out_of_range():
    df = clean()
    df.loc[50, "egt"] = np.nan
    res = validate(df, {"time_column": "timestamp"})
    assert "OUT_OF_RANGE" not in codes(res)
    assert "MISSING_VALUES" in codes(res) and "GAP_FILLED" in codes(res)


def test_edge_gaps_not_filled():
    df = clean()
    df.loc[0:1, "cht"] = np.nan            # leading gap has nothing to interpolate from
    res = validate(df, {"time_column": "timestamp"})
    assert len(res.segments) == 1 and len(res.segments[0]) == 298


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)