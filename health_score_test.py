import pandas as pd
import numpy as np

# Healthy operating ranges from schema.py
HEALTHY_RANGES = {
    "rpm": (4800, 5200),
    "egt": (680, 760),
    "cht": (150, 190),
    "oil_pressure": (55, 75),
    "oil_temp": (85, 110),
    "vibration": (0.5, 2.0),
    "fuel_flow": (18, 24),
}

WEIGHTS = {
    "rpm": 0.10,
    "egt": 0.20,
    "cht": 0.15,
    "oil_pressure": 0.15,
    "oil_temp": 0.10,
    "vibration": 0.15,
    "fuel_flow": 0.15,
}


def sensor_health(value, lower, upper):
    """Return a health value between 0 and 1."""

    if lower <= value <= upper:
        return 1.0

    if value < lower:
        score = 1 - (lower - value) / lower
    else:
        score = 1 - (value - upper) / upper

    return np.clip(score, 0.0, 1.0)


def calculate_health_score(row):
    score = 0.0

    for sensor, (lower, upper) in HEALTHY_RANGES.items():
        h = sensor_health(row[sensor], lower, upper)
        score += WEIGHTS[sensor] * h

    return score * 100


# Load one real flight
file_path = "data/raw/cooling_degradation_000.csv"
df = pd.read_csv(file_path)

# Calculate health score
df["health_score"] = df.apply(calculate_health_score, axis=1)

print("\n=== HEALTH SCORE TEST ===")

print("\nFirst 5 rows:")
print(df[["timestamp", "fault_type", "fault_onset_idx", "health_score"]].head())

print("\nHealth score BEFORE fault:")
before = df[df.index < df["fault_onset_idx"].iloc[0]]
print(before["health_score"].describe())

print("\nHealth score AFTER fault:")
after = df[df.index >= df["fault_onset_idx"].iloc[0]]
print(after["health_score"].describe())

print("\nOverall:")
print(df["health_score"].describe())

print("\nLowest 10 health scores:")
print(
    df[
        ["timestamp", "cht", "egt", "health_score", "fault_type"]
    ]
    .sort_values("health_score")
    .head(10)
)