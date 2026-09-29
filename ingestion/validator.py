"""
ingestion/validator.py -- the boundary between raw engine data and the AeroTwin pipeline.

Turns a raw table (CSV export, ground-test-cell log, telemetry dump) into clean,
uniformly sampled segments in the pipeline schema:
    timestamp (int tick 0..n-1), rpm, egt, cht, oil_pressure, oil_temp, vibration, fuel_flow

It never invents data. It maps column names, converts units, fixes ordering and
sampling, fills only SHORT gaps by interpolation, blanks implausible values, and
reports everything it did. If required sensors are missing it rejects, and it never
fills a missing sensor with fake values.

CLI (from the repo root):
    python -m ingestion.validator raw.csv --config ingestion/engine_config.example.json --out-dir ingest_out
"""
import argparse
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
TIME_NAMES = ("timestamp", "time", "t", "time_s", "seconds", "datetime")

SENSORS = ["rpm", "egt", "cht", "oil_pressure", "oil_temp", "vibration", "fuel_flow"]

# Units are engine-specific, so only sign limits are applied by default.
# Put real physical limits per sensor in the engine config.
DEFAULT_LIMITS = {
    "rpm": (0.0, None),
    "oil_pressure": (0.0, None),
    "vibration": (0.0, None),
    "fuel_flow": (0.0, None),
}

DEFAULT_CONFIG = {
    "target_dt_s": 1.0,        # seconds per tick the pipeline expects
    "dt_tolerance": 0.10,      # allowed relative deviation of median dt before resampling is flagged
    "max_gap_ticks": 3,        # gaps up to this long are linearly interpolated
    "min_segment_ticks": 30,   # shortest usable segment (the classifier window length)
    "min_usable_fraction": 0.5,
    "time_column": None,
    "sensors": {},
}


@dataclass
class Issue:
    level: str          # "INFO" | "WARN" | "REJECT"
    code: str
    detail: str
    sensor: str = ""


@dataclass
class IngestResult:
    status: str = "OK"                      # OK | WARN | REJECT
    issues: list = field(default_factory=list)
    segments: list = field(default_factory=list)   # list of DataFrames, pipeline schema
    segment_info: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def add(self, level, code, detail, sensor=""):
        self.issues.append(Issue(level, code, detail, sensor))

    def finalize(self):
        levels = {i.level for i in self.issues}
        if "REJECT" in levels or not self.segments:
            self.status = "REJECT"
        elif "WARN" in levels:
            self.status = "WARN"
        else:
            self.status = "OK"
        return self

    def report(self):
        return {
            "status": self.status,
            "stats": self.stats,
            "segments": self.segment_info,
            "issues": [i.__dict__ for i in self.issues],
        }


def load_config(path):
    cfg = dict(DEFAULT_CONFIG)
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            cfg.update(json.load(fh))
    return cfg


def _sensor_spec(cfg, name):
    spec = dict(cfg["sensors"].get(name, {}))
    spec.setdefault("column", name)
    spec.setdefault("scale", 1.0)
    spec.setdefault("offset", 0.0)
    if "limits" not in spec:
        lo, hi = DEFAULT_LIMITS.get(name, (None, None))
        spec["limits"] = [lo, hi]
    return spec


def _fill_short_gaps(col: pd.Series, max_gap: int) -> pd.Series:
    """Linearly interpolate runs of NaN that are <= max_gap long and have valid
    samples on both sides. Longer runs, and runs at the very start or end, stay NaN."""
    isna = col.isna().values
    full = col.interpolate(method="linear", limit_area="inside").values
    out = col.values.copy()
    n, i = len(col), 0
    while i < n:
        if not isna[i]:
            i += 1
            continue
        j = i
        while j < n and isna[j]:
            j += 1
        if i > 0 and j < n and (j - i) <= max_gap:
            out[i:j] = full[i:j]
        i = j
    return pd.Series(out, index=col.index)


def validate(raw: pd.DataFrame, cfg=None) -> IngestResult:
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    res = IngestResult()
    dt = float(cfg["target_dt_s"])
    res.stats["raw_rows"] = int(len(raw))

    if len(raw) == 0:
        res.add("REJECT", "EMPTY", "input table has no rows")
        return res.finalize()

    # 1. map columns; missing sensors are a hard reject (never filled)
    specs, missing = {}, []
    for s in SENSORS:
        spec = _sensor_spec(cfg, s)
        specs[s] = spec
        if spec["column"] not in raw.columns:
            missing.append(s)
            res.add("REJECT", "MISSING_SENSOR",
                    f"column '{spec['column']}' not found; the pipeline needs all 7 sensors", s)
    if missing:
        res.stats["missing_sensors"] = missing
        return res.finalize()

    # 2. time axis
    tcol = cfg.get("time_column")
    explicit = bool(tcol)
    if not tcol:
        lower = {str(c).strip().lower(): c for c in raw.columns}
        for cand in TIME_NAMES:
            if cand in lower:
                tcol = lower[cand]
                break
    t = None
    if tcol and tcol in raw.columns:
        t = pd.to_numeric(raw[tcol], errors="coerce")
        if t.isna().all():
            t = pd.to_datetime(raw[tcol], errors="coerce")
            if t.isna().all():
                t = None
            else:
                t = (t - t.min()).dt.total_seconds()
        if t is None:
            if explicit:
                res.add("REJECT", "BAD_TIME", f"cannot parse time column '{tcol}'")
                return res.finalize()
        else:
            t = t.astype(float)
            if not explicit:
                res.add("INFO", "TIME_COLUMN_DETECTED",
                        f"using column '{tcol}' as the time axis")
    if t is None:
        res.add("WARN", "NO_TIME_COLUMN",
                f"no time column; assuming 1 row = {dt} s in the given order")
        t = pd.Series(np.arange(len(raw)) * dt, index=raw.index, dtype=float)

    df = pd.DataFrame({"t": t.values})
    for s in SENSORS:
        spec = specs[s]
        col = pd.to_numeric(raw[spec["column"]], errors="coerce").astype(float)
        bad = int(col.isna().sum() - raw[spec["column"]].isna().sum())
        if bad > 0:
            res.add("WARN", "NON_NUMERIC", f"{bad} non-numeric values treated as missing", s)
        df[s] = col.values * float(spec["scale"]) + float(spec["offset"])

    df = df[df["t"].notna()]
    if not df["t"].is_monotonic_increasing:
        res.add("WARN", "OUT_OF_ORDER", "timestamps were not in order; sorted")
        df = df.sort_values("t")
    dups = int(df["t"].duplicated().sum())
    if dups:
        res.add("WARN", "DUPLICATE_TIMESTAMPS", f"{dups} duplicate timestamps averaged")
        df = df.groupby("t", as_index=False).mean()

    # 3. implausible values -> missing (counted, never silently clipped)
    for s in SENSORS:
        lo, hi = specs[s]["limits"]
        n_missing = int(df[s].isna().sum())
        if n_missing:
            res.add("INFO", "MISSING_VALUES",
                    f"{n_missing} values already missing in the input "
                    f"(includes any non-numeric entries)", s)
        mask = pd.Series(False, index=df.index)
        if lo is not None:
            mask |= df[s] < lo
        if hi is not None:
            mask |= df[s] > hi
        mask |= np.isinf(df[s])
        n = int(mask.sum())
        if n:
            res.add("WARN", "OUT_OF_RANGE",
                    f"{n} values outside limits [{lo}, {hi}] or infinite, blanked", s)
            df.loc[mask, s] = np.nan

    # 4. sampling: report actual rate, place on the uniform grid
    if len(df) < 2:
        res.add("REJECT", "TOO_SHORT", "fewer than 2 valid timestamps")
        return res.finalize()
    med_dt = float(np.median(np.diff(df["t"].values)))
    res.stats["median_dt_s"] = round(med_dt, 4)
    if med_dt <= 0:
        res.add("REJECT", "BAD_TIME", "non-positive median time step")
        return res.finalize()
    rel = abs(med_dt - dt) / dt
    if rel > cfg["dt_tolerance"]:
        kind = "UPSAMPLED" if med_dt > dt else "DOWNSAMPLED"
        res.add("WARN", "RESAMPLED",
                f"median step {med_dt:.3f}s vs target {dt}s ({kind}); "
                + ("interpolated values are not measurements"
                   if med_dt > dt else "samples were averaged"))

    k = np.round((df["t"].values - df["t"].values[0]) / dt).astype(int)
    grid = df[SENSORS].copy()
    grid["k"] = k
    grid = grid.groupby("k").mean()
    grid = grid.reindex(np.arange(0, int(k.max()) + 1))
    n_grid = len(grid)
    res.stats["grid_ticks"] = int(n_grid)

    # 5. fill only gaps that are short in their entirety (a long gap is left untouched,
    #    never partially filled from its edges)
    before = grid.isna().sum()
    grid = grid.apply(lambda c: _fill_short_gaps(c, int(cfg["max_gap_ticks"])))
    after = grid.isna().sum()
    for s in SENSORS:
        filled = int(before[s] - after[s])
        left = int(after[s])
        if filled:
            res.add("INFO", "GAP_FILLED",
                    f"{filled} ticks interpolated (gaps <= {cfg['max_gap_ticks']} ticks)", s)
        if left:
            res.add("WARN", "LONG_GAP", f"{left} ticks missing in gaps too long to fill", s)

    # 6. whole-file frozen sensor check (fine-grained checks stay in sensor_trust)
    for s in SENSORS:
        v = grid[s].dropna()
        if len(v) >= 30 and float(v.std()) == 0.0:
            res.add("WARN", "FLAT_SENSOR", "sensor is constant across the whole input", s)

    # 7. cut into contiguous fully-valid segments
    valid = grid.notna().all(axis=1).values
    min_len = int(cfg["min_segment_ticks"])
    usable = 0
    i = 0
    while i < n_grid:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j < n_grid and valid[j]:
            j += 1
        if j - i >= min_len:
            seg = grid.iloc[i:j].reset_index(drop=True)
            seg.insert(0, "timestamp", np.arange(len(seg), dtype=int))
            res.segments.append(seg)
            res.segment_info.append({
                "start_time_s": round(float(df["t"].values[0]) + i * dt, 3),
                "ticks": int(j - i),
            })
            usable += j - i
        i = j

    frac = usable / n_grid if n_grid else 0.0
    res.stats["usable_fraction"] = round(frac, 4)
    res.stats["segments"] = len(res.segments)
    if frac < cfg["min_usable_fraction"]:
        res.add("REJECT", "LOW_USABLE",
                f"only {frac:.0%} of the input is usable (need {cfg['min_usable_fraction']:.0%})")
    elif frac < 1.0:
        res.add("WARN", "PARTIAL_USABLE", f"{frac:.0%} of the input is usable after cleaning")
    return res.finalize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default="ingest_out")
    args = ap.parse_args()

    res = validate(pd.read_csv(args.csv), load_config(args.config))
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "ingest_report.json"), "w", encoding="utf-8") as fh:
        json.dump(res.report(), fh, indent=2)
    for n, seg in enumerate(res.segments):
        seg.to_csv(os.path.join(args.out_dir, f"segment_{n:02d}.csv"), index=False)

    print(f"status: {res.status}")
    for i in res.issues:
        print(f"  [{i.level}] {i.code} {i.sensor}: {i.detail}")
    print(f"segments written: {len(res.segments)} -> {args.out_dir}")


if __name__ == "__main__":
    main()