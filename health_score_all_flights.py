import pandas as pd
import glob
import os
import numpy as np

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
        health = sensor_health(
            row[sensor],
            lower,
            upper
        )
        score += WEIGHTS[sensor] * health

    return score * 100


# Find all generated flights
files = glob.glob("data/raw/*.csv")

print(f"Testing {len(files)} flights...\n")

results = []

for file in files:

    df = pd.read_csv(file)

    # Calculate health score for every timestep
    df["health_score"] = df.apply(
        calculate_health_score,
        axis=1
    )

    fault = df["fault_type"].iloc[-1]
    onset = df["fault_onset_idx"].iloc[0]

    if onset >= 0:

        before = df.loc[
            df.index < onset,
            "health_score"
        ].mean()

        after = df.loc[
            df.index >= onset,
            "health_score"
        ].mean()

    else:
        # Healthy flights have no fault onset
        before = df["health_score"].mean()
        after = df["health_score"].mean()

    results.append({
        "flight": os.path.basename(file),
        "fault": fault,
        "before": before,
        "after": after
    })


results_df = pd.DataFrame(results)

summary = (
    results_df
    .groupby("fault")[["before", "after"]]
    .mean()
    .round(2)
)

print("=== HEALTH SCORE SUMMARY ===\n")
print(summary)

print("\n=== SCORE DROP ===\n")

summary["drop"] = (
    summary["before"] - summary["after"]
).round(2)

print(summary)