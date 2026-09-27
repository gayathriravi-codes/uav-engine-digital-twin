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


NOMINAL_COUPLINGS = {
    "egt": 0.6,
    "cht": 0.4,
    "oil_pressure": 0.3,
    "oil_temp": 0.2,
    "vibration": 0.3,
    "fuel_flow": 0.7,
}

NOMINAL_NOISE_FRAC = {
    "egt": 0.08,
    "cht": 0.08,
    "oil_pressure": 0.08,
    "oil_temp": 0.08,
    "vibration": 0.15,
    "fuel_flow": 0.08,
}


def simulate_healthy_flight(duration_timesteps=300, seed=None, coupling_overrides=None,
                             noise_frac_overrides=None):
    rng = np.random.default_rng(seed)
    t = np.arange(duration_timesteps)

    couplings = dict(NOMINAL_COUPLINGS)
    if coupling_overrides:
        couplings.update(coupling_overrides)
    noise_fracs = dict(NOMINAL_NOISE_FRAC)
    if noise_frac_overrides:
        noise_fracs.update(noise_frac_overrides)

    rpm_mid, rpm_span = _mid("rpm"), _span("rpm")
    rpm_walk = np.cumsum(rng.normal(0, rpm_span * 0.02, duration_timesteps))
    rpm = np.clip(rpm_mid + rpm_walk, rpm_mid - rpm_span, rpm_mid + rpm_span)
    rpm_dev = (rpm - rpm_mid) / rpm_span

    def correlated(field):
        mid, span = _mid(field), _span(field)
        base = mid + couplings[field] * rpm_dev * span
        noise = rng.normal(0, span * noise_fracs[field], duration_timesteps)
        return np.clip(base + noise, mid - span * 1.2, mid + span * 1.2)

    egt = correlated("egt")
    cht = correlated("cht")
    oil_pressure = correlated("oil_pressure")
    oil_temp = correlated("oil_temp")
    vibration = correlated("vibration")
    fuel_flow = correlated("fuel_flow")

    df = pd.DataFrame({
        "timestamp": t, "rpm": rpm, "egt": egt, "cht": cht,
        "oil_pressure": oil_pressure, "oil_temp": oil_temp,
        "vibration": vibration, "fuel_flow": fuel_flow,
    })
    return df


if __name__ == "__main__":
    df = simulate_healthy_flight(seed=42)
    print(df.describe())
    print(df.head())
    df_default = simulate_healthy_flight(seed=42)
    df_explicit_nominal = simulate_healthy_flight(seed=42, coupling_overrides=NOMINAL_COUPLINGS,
                                                   noise_frac_overrides=NOMINAL_NOISE_FRAC)
    assert df_default.equals(df_explicit_nominal), "Nominal behavior changed"
    print("[OK] Nominal output unchanged.")
