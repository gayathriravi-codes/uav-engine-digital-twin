"""
mission_profiles.py
----------------------
Environmental/mission-profile parameterization layer for simulate_healthy_flight,
added to satisfy PS 26054 Section E (Simulation & Replay Capability): "Engine
behavior simulation during High Altitude / Endurance mission / Hot-weather
operation / Rapid throttle transitions".

DELIBERATELY ADDITIVE: simulate_healthy_flight() in simulate_healthy.py is
UNCHANGED and untouched -- every existing fault injector, the 108+123-flight
dataset, the RUL ensemble, and the leak-fixed classifier all depend on its
current exact behavior. This module wraps it with a new function that
defaults to reproducing that same behavior byte-for-byte when no profile is
given (profile=None), so nothing existing needs to be regenerated or retrained
to add this capability.

Physics basis (documented here so it can be defended, not just asserted):
  - Altitude -> air density ratio via the ICAO standard atmosphere
    approximation: sigma = (1 - 2.2557e-5 * altitude_m) ** 4.2561, valid
    below the tropopause (~11km), which covers all MALE UAV operating
    altitudes. Lower sigma -> less cooling airflow + leaner effective
    mixture for a fixed throttle setting on a naturally-aspirated piston
    engine -> raises EGT and CHT baselines. This is standard IC-engine
    behavior (see e.g. Heywood, "Internal Combustion Engine Fundamentals"),
    not a fitted/invented curve.
  - Hot weather -> ambient temperature raises oil_temp and CHT baselines
    roughly in proportion to the reduced temperature differential available
    for convective cooling.
  - Rapid throttle transitions -> RPM follows a step/ramp profile with fast
    transitions instead of the default slow random walk, which is a
    genuine stress test of the EXISTING sensor_trust.py MAX_RATE_OF_CHANGE
    thresholds -- this scenario exercises a real part of the system rather
    than adding a new one.
  - Endurance mission -> identical physics, just a longer duration_timesteps;
    ties directly into the existing RUL degradation curve machinery.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import HEALTHY_RANGES
from simulate_healthy import simulate_healthy_flight


def _mid(field):
    lo, hi = HEALTHY_RANGES[field]
    return (lo + hi) / 2


def _span(field):
    lo, hi = HEALTHY_RANGES[field]
    return (hi - lo) / 2


def air_density_ratio(altitude_m):
    """ICAO standard atmosphere density ratio (sigma), valid below ~11km."""
    return max(0.05, (1 - 2.2557e-5 * altitude_m) ** 4.2561)


# Named mission profiles, matching PS 26054 Section E's explicit list.
# Each is a dict of environmental parameters consumed by
# simulate_mission_flight() below. "nominal" reproduces the original
# simulate_healthy_flight() behavior exactly (altitude=0, ambient=default,
# no throttle transients).
MISSION_PROFILES = {
    "nominal": {
        "altitude_m": 0,
        "ambient_temp_offset_c": 0,
        "throttle_transient": False,
        "description": "Sea-level, standard-day baseline (matches original simulate_healthy_flight).",
    },
    "high_altitude": {
        "altitude_m": 6000,
        "ambient_temp_offset_c": -10,  # standard lapse rate, colder at altitude
        "throttle_transient": False,
        "description": "MALE UAV cruise altitude (~20,000 ft): reduced air density, colder ambient.",
    },
    "endurance": {
        "altitude_m": 4000,
        "ambient_temp_offset_c": -5,
        "throttle_transient": False,
        "duration_multiplier": 4.0,  # long-endurance ISR-style mission
        "description": "Extended-duration cruise mission -- same physics, longer flight.",
    },
    "hot_weather": {
        "altitude_m": 500,
        "ambient_temp_offset_c": +20,  # hot-day ground/low-altitude ops
        "throttle_transient": False,
        "description": "Hot-day ground/low-altitude operation: elevated ambient temperature.",
    },
    "rapid_throttle": {
        "altitude_m": 1000,
        "ambient_temp_offset_c": 0,
        "throttle_transient": True,
        "description": "Frequent rapid throttle changes -- e.g. terrain-following, evasive maneuvering.",
    },
}


def _apply_environmental_shift(df, altitude_m, ambient_temp_offset_c):
    """
    Shifts EGT/CHT/oil_temp baselines according to altitude (via air
    density ratio) and ambient temperature offset. Applied AFTER the
    normal healthy-flight generation, as an environmental bias layered
    on top of the existing RPM-coupled sensor model -- so all existing
    sensor correlation logic in simulate_healthy_flight is reused, not
    reimplemented.
    """
    sigma = air_density_ratio(altitude_m)
    density_deficit = 1.0 - sigma  # 0 at sea level, grows with altitude

    # EGT and CHT rise as air density drops (leaner effective mixture,
    # less cooling airflow) -- magnitude scaled against each sensor's own
    # healthy-range span so the effect is proportionate across engines
    # with different absolute sensor ranges.
    df["egt"] = df["egt"] + density_deficit * _span("egt") * 0.9
    df["cht"] = df["cht"] + density_deficit * _span("cht") * 0.7

    # Ambient temperature offset shifts oil_temp and CHT baselines
    # roughly in proportion to the reduced cooling differential.
    df["oil_temp"] = df["oil_temp"] + ambient_temp_offset_c * 0.6
    df["cht"] = df["cht"] + ambient_temp_offset_c * 0.3

    return df


def _apply_throttle_transient(df, rng):
    """
    Replaces the smooth RPM random walk with a step/ramp profile
    containing several rapid transitions, and re-derives fuel_flow's
    RPM-coupled component to follow the new RPM trace (since fuel_flow
    is naturally driven by throttle position, not by CHT/EGT/etc.).

    This is a genuine input-level change (not a cosmetic label): it
    directly stress-tests sensor_trust.py's MAX_RATE_OF_CHANGE checks,
    which are tuned against the ORIGINAL smooth RPM walk and may or may
    not correctly tolerate legitimate rapid throttle changes vs. flagging
    them as sensor glitches -- worth reporting honestly either way.
    """
    n = len(df)
    rpm_mid, rpm_span = _mid("rpm"), _span("rpm")

    n_transients = max(3, n // 60)
    transition_points = np.sort(rng.choice(np.arange(20, n - 20), size=n_transients, replace=False))

    rpm = np.full(n, rpm_mid)
    current_level = rpm_mid
    last_point = 0
    for point in transition_points:
        new_level = rpm_mid + rng.uniform(-1, 1) * rpm_span * rng.uniform(0.5, 1.0)
        ramp_len = rng.integers(2, 6)  # fast transition, 2-5 timesteps
        ramp_end = min(point + ramp_len, n)
        rpm[last_point:point] = current_level
        rpm[point:ramp_end] = np.linspace(current_level, new_level, ramp_end - point)
        current_level = new_level
        last_point = ramp_end
    rpm[last_point:] = current_level
    rpm += rng.normal(0, rpm_span * 0.02, n)  # normal sensor noise on top

    df["rpm"] = rpm
    rpm_dev = (rpm - rpm_mid) / rpm_span

    # Re-derive fuel_flow to follow the new RPM trace directly, since
    # fuel_flow should track throttle/RPM tightly, not the old smooth walk.
    mid, span = _mid("fuel_flow"), _span("fuel_flow")
    df["fuel_flow"] = np.clip(
        mid + 0.7 * rpm_dev * span + rng.normal(0, span * 0.08, n),
        mid - span * 1.2, mid + span * 1.2
    )

    return df


def simulate_mission_flight(profile_name="nominal", duration_timesteps=300, seed=None):
    """
    Generates a healthy flight under a named mission profile.

    profile_name="nominal" reproduces simulate_healthy_flight()'s original
    behavior exactly (no environmental shift, no throttle transient) --
    existing code calling simulate_healthy_flight() directly is completely
    unaffected by this module's existence.

    Returns the same DataFrame shape/columns as simulate_healthy_flight,
    plus a "mission_profile" column recording which profile generated it
    (so downstream fault injection / analysis can condition on it if useful).
    """
    if profile_name not in MISSION_PROFILES:
        raise ValueError(f"Unknown mission profile '{profile_name}'. Options: {list(MISSION_PROFILES)}")

    profile = MISSION_PROFILES[profile_name]
    rng = np.random.default_rng(seed)

    actual_duration = int(duration_timesteps * profile.get("duration_multiplier", 1.0))
    df = simulate_healthy_flight(duration_timesteps=actual_duration, seed=seed)

    if profile["altitude_m"] != 0 or profile["ambient_temp_offset_c"] != 0:
        df = _apply_environmental_shift(df, profile["altitude_m"], profile["ambient_temp_offset_c"])

    if profile.get("throttle_transient"):
        df = _apply_throttle_transient(df, rng)

    df["mission_profile"] = profile_name
    df["altitude_m"] = profile["altitude_m"]
    df["ambient_temp_offset_c"] = profile["ambient_temp_offset_c"]

    return df


if __name__ == "__main__":
    import pandas as pd
    print("Sanity check: comparing all mission profiles at seed=42\n")
    for name in MISSION_PROFILES:
        df = simulate_mission_flight(profile_name=name, duration_timesteps=300, seed=42)
        print(f"--- {name} ---")
        print(f"  {MISSION_PROFILES[name]['description']}")
        print(f"  duration: {len(df)} timesteps")
        print(f"  egt mean: {df['egt'].mean():.1f}  cht mean: {df['cht'].mean():.1f}  "
              f"oil_temp mean: {df['oil_temp'].mean():.1f}")
        if name == "rapid_throttle":
            rpm_diffs = df["rpm"].diff().abs()
            print(f"  max single-step RPM change: {rpm_diffs.max():.1f} "
                  f"(sensor_trust.py MAX_RATE_OF_CHANGE['rpm'] = 500)")
        print()

    # Confirm nominal profile reproduces the original function exactly
    from simulate_healthy import simulate_healthy_flight as original
    original_df = original(duration_timesteps=300, seed=42)
    nominal_df = simulate_mission_flight(profile_name="nominal", duration_timesteps=300, seed=42)
    shared_cols = [c for c in original_df.columns if c in nominal_df.columns]
    matches = original_df[shared_cols].equals(nominal_df[shared_cols])
    print(f"Nominal profile matches original simulate_healthy_flight exactly: {matches}")
    assert matches, "REGRESSION: nominal profile must exactly reproduce original behavior!"