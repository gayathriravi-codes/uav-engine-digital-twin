import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import HEALTHY_RANGES


def _mid(field):
    lo, hi = HEALTHY_RANGES[field]
    return (lo + hi) / 2


def _span(field):
    lo, hi = HEALTHY_RANGES[field]
    return (hi - lo) / 2


def simulate_healthy_flight(duration_timesteps=300, seed=None):
    rng = np.random.default_rng(seed)
    t = np.arange(duration_timesteps)

    rpm_mid, rpm_span = _mid("rpm"), _span("rpm")
    rpm_walk = np.cumsum(rng.normal(0, rpm_span * 0.02, duration_timesteps))
    rpm = np.clip(rpm_mid + rpm_walk, rpm_mid - rpm_span, rpm_mid + rpm_span)
    rpm_dev = (rpm - rpm_mid) / rpm_span

    def correlated(field, rpm_coupling, noise_frac=0.08):
        mid, span = _mid(field), _span(field)
        base = mid + rpm_coupling * rpm_dev * span
        noise = rng.normal(0, span * noise_frac, duration_timesteps)
        return np.clip(base + noise, mid - span * 1.2, mid + span * 1.2)

    egt = correlated("egt", rpm_coupling=0.6)
    cht = correlated("cht", rpm_coupling=0.4)
    oil_pressure = correlated("oil_pressure", rpm_coupling=0.3)
    oil_temp = correlated("oil_temp", rpm_coupling=0.2)
    vibration = correlated("vibration", rpm_coupling=0.3, noise_frac=0.15)
    fuel_flow = correlated("fuel_flow", rpm_coupling=0.7)

    df = pd.DataFrame({
        "timestamp": t, "rpm": rpm, "egt": egt, "cht": cht,
        "oil_pressure": oil_pressure, "oil_temp": oil_temp,
        "vibration": vibration, "fuel_flow": fuel_flow,
    })
    return df


if __name__ == "__main__":
    df = simulate_healthy_flight(seed=42)
    print(df.describe())
    print("\nFirst 5 rows:")
    print(df.head())