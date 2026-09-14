import glob
import pandas as pd
import numpy as np
from models.classification.feature_extraction import calculate_health_score

RUL_BUCKETS = [(0, 10), (10, 30), (30, 60), (60, 150), (150, 1000)]

rows = []
for path in sorted(glob.glob("data/raw/*.csv")):
    df = pd.read_csv(path).sort_values("timestamp").reset_index(drop=True)

    # Skip genuinely healthy-only flights: true_rul_timesteps stays -1
    # throughout (not to be confused with fault_type=="none" ROWS, which
    # every fault flight also has before fault_onset_idx).
    if (df["true_rul_timesteps"] == -1).all():
        continue

    scores = calculate_health_score(df)
    for i in range(len(df)):
        rows.append({
            "fault_type": df["fault_type"].iloc[i],
            "true_rul": df["true_rul_timesteps"].iloc[i],
            "health_score": scores.iloc[i],
        })

data = pd.DataFrame(rows)
print(f"Total rows collected: {len(data)}\n")

print("Health score by RUL bucket (all fault types combined):")
for lo, hi in RUL_BUCKETS:
    bucket = data[(data["true_rul"] >= lo) & (data["true_rul"] < hi)]
    if len(bucket) == 0:
        continue
    print(f"  RUL [{lo:4d},{hi:4d}): n={len(bucket):5d}  "
          f"mean={bucket['health_score'].mean():6.2f}  "
          f"min={bucket['health_score'].min():6.2f}  "
          f"max={bucket['health_score'].max():6.2f}")

print("\nSame, broken down by fault type, RUL < 10 only (should mostly be CRITICAL):")
near_fail = data[data["true_rul"] < 10]
print(near_fail.groupby("fault_type")["health_score"].agg(["mean", "min", "max", "count"]))