"""Fault x detection-layer coverage matrix, built only from existing docs/ results."""
import argparse
import json
import os

import numpy as np
import pandas as pd

DOCS = "docs"

# Physics-residual layer: qualitative, taken from the earlier validation session
# (physics_residual.py). NOT recomputed here.
PHYSICS = {
    "overheat": "detects (10-40x residual spike)",
    "misfire": "detects (100% det, 1.73% FP)",
    "sensor_drift": "no signal (structural blind spot)",
}


def load_csv(name):
    p = os.path.join(DOCS, name)
    if not os.path.exists(p):
        raise SystemExit(f"missing file: {p} (run from the project root)")
    return pd.read_csv(p)


def load_json(name):
    p = os.path.join(DOCS, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def nz(x):
    return 0.0 if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def levels(df, col):
    """Rate at lowest / middle / highest severity above 0 (0 = no fault injected)."""
    if df is not None:
        df = df[df["severity"] > 0]
    if df is None or df.empty:
        return [np.nan] * 3, [np.nan] * 3
    g = df.groupby("severity")[col].mean().sort_index()
    sev, val = list(g.index), list(g.values)
    ix = [0, len(sev) // 2, len(sev) - 1]
    return [float(val[i]) for i in ix], [float(sev[i]) for i in ix]


def at(df, sevs, col):
    if df is None or df.empty:
        return [np.nan] * len(sevs)
    g = df.groupby(df["severity"].round(3))[col].mean()
    return [float(g.get(round(float(s), 3), np.nan)) for s in sevs]


def verdict(c, t):
    hi_c, hi_t = nz(c[2]), nz(t[2])
    lo = max(nz(c[0]), nz(t[0]))
    hi = max(hi_c, hi_t)
    layer = "classifier" if hi_c >= hi_t else "sensor_trust"
    if hi >= 0.9 and lo >= 0.9:
        return f"covered ({layer})"
    if hi >= 0.9:
        return f"covered at high severity only ({layer})"
    if hi >= 0.5:
        return f"PARTIAL ({layer}, best {hi:.2f})"
    return f"BLIND SPOT (best {hi:.2f})"


def make_row(item, clf_df, trust_df, phys):
    c, sev = levels(clf_df, "rate")
    t = at(trust_df, sev, "rate_target") if trust_df is not None else [np.nan] * 3
    return {
        "item": item,
        "severity_low_mid_high": "/".join("-" if np.isnan(s) else f"{s:g}" for s in sev),
        "clf_low": c[0], "clf_mid": c[1], "clf_high": c[2],
        "trust_low": t[0], "trust_mid": t[1], "trust_high": t[2],
        "physics": phys,
        "best_low": max(nz(c[0]), nz(t[0])),
        "best_high": max(nz(c[2]), nz(t[2])),
        "verdict": verdict(c, t),
    }


def build(noise):
    pod = load_csv("pod_results.csv")
    strata = load_csv("pod_drift_strata.csv")
    trust = load_csv("trust_check_drift.csv")

    print("faults in pod_results:", sorted(pod["fault"].unique()))
    print("noise levels in pod_results:", sorted(pod["noise"].unique()))
    print("sub_types in pod_drift_strata:", sorted(strata["sub_type"].unique()))
    print("sub_types in trust_check_drift:", sorted(trust["sub_type"].unique()))
    print()

    rows = []
    pn = pod[np.isclose(pod["noise"], noise)]
    for fault in sorted(pn["fault"].unique()):
        if fault in ("none", "sensor_drift"):
            continue
        d = pn[pn["fault"] == fault]
        rows.append(make_row(fault, d, None, PHYSICS.get(fault, "not measured")))

    for (sub, tgt), d in strata.groupby(["sub_type", "target"]):
        tr = trust[(trust["sub_type"] == sub) & (trust["target"] == tgt)]
        rows.append(make_row(f"sensor_drift: {sub} / {tgt}" + (" [0% noise data]" if noise > 0 else ""), d,
                             tr if not tr.empty else None, PHYSICS["sensor_drift"]))
    return pd.DataFrame(rows)


def false_alarms():
    out = []
    for name, fname in (("classifier", "pod_healthy_fa.json"),
                        ("sensor_trust", "trust_check_healthy.json")):
        data = load_json(fname)
        if not data:
            continue
        for r in data:
            out.append({"layer": name, "noise": r["noise"], "flights": r["flights"],
                        "false_alarm_flights": r["alarm_flights"],
                        "rate": round(r["flight_rate"], 4),
                        "ci95": f"{r['ci95_low']:.3f}-{r['ci95_high']:.3f}"})
    return pd.DataFrame(out)


def save_png(df, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping PNG (python -m pip install matplotlib)")
        return
    cols = ["item", "clf_low", "clf_mid", "clf_high",
            "trust_low", "trust_mid", "trust_high", "physics", "verdict"]
    rate_cols = cols[1:7]
    cell = [[("" if (isinstance(v, float) and np.isnan(v)) else
              (f"{v:.2f}" if c in rate_cols else str(v)))
             for c, v in zip(cols, r)] for r in df[cols].values.tolist()]
    fig, ax = plt.subplots(figsize=(17, 0.55 * len(df) + 1.8))
    ax.axis("off")
    tb = ax.table(cellText=cell, colLabels=cols, loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(8)
    tb.auto_set_column_width(list(range(len(cols))))
    for i, r in enumerate(df[cols].values.tolist(), start=1):
        for j, c in enumerate(cols):
            v = r[j]
            if c in rate_cols and not (isinstance(v, float) and np.isnan(v)):
                tb[i, j].set_facecolor(plt.cm.RdYlGn(float(v)))
        vt = r[-1]
        tb[i, len(cols) - 1].set_facecolor(
            "#f4b6b6" if vt.startswith("BLIND") else
            "#ffe08a" if vt.startswith("PARTIAL") or "only" in vt else "#b8e0b8")
    ax.set_title("AeroTwin detection coverage: fault x layer (detection rate, low/mid/high severity)",
                 fontsize=11)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=0.0)
    a = ap.parse_args()
    TAG = f"_n{int(round(a.noise * 100))}"

    df = build(a.noise)
    fa = false_alarms()

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 40)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nHealthy-flight false alarms:")
    print(fa.to_string(index=False))

    df.to_csv(os.path.join(DOCS, f"coverage_matrix{TAG}.csv"), index=False)
    fa.to_csv(os.path.join(DOCS, "coverage_false_alarms.csv"), index=False)
    print(f"\nsaved docs/coverage_matrix{TAG}.csv, docs/coverage_false_alarms.csv")
    save_png(df, os.path.join(DOCS, f"coverage_matrix{TAG}.png"))


if __name__ == "__main__":
    main()