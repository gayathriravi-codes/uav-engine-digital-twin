import pandas as pd
import glob

SENSORS = [
    "rpm",
    "egt",
    "cht",
    "oil_pressure",
    "oil_temp",
    "vibration",
    "fuel_flow",
]

files = glob.glob("data/raw/*.csv")

results = []

for file in files:

    df = pd.read_csv(file)

    fault = df["fault_type"].iloc[-1]
    onset = df["fault_onset_idx"].iloc[0]

    if onset < 0:
        continue

    before = df.loc[df.index < onset, SENSORS].mean()
    after = df.loc[df.index >= onset, SENSORS].mean()

    for sensor in SENSORS:

        change = after[sensor] - before[sensor]

        results.append({
            "fault": fault,
            "sensor": sensor,
            "before": before[sensor],
            "after": after[sensor],
            "change": change,
        })


result_df = pd.DataFrame(results)

summary = (
    result_df
    .groupby(["fault", "sensor"])[["before", "after", "change"]]
    .mean()
    .round(2)
)

print("\n=== SENSOR BEHAVIOUR BY FAULT ===\n")
print(summary.to_string())