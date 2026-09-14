import pandas as pd
import glob
from models.classification.feature_extraction import calculate_health_score

rows = []
for path in sorted(glob.glob("data/raw/*.csv")):
    df = pd.read_csv(path).sort_values("timestamp").reset_index(drop=True)
    if (df["true_rul_timesteps"] == -1).all():
        continue
    scores = calculate_health_score(df)
    for i in range(len(df)):
        rows.append({
            "fault_type": df["fault_type"].iloc[i],
            "true_rul": df["true_rul_timesteps"].iloc[i],
            "health_score": 0.7 * scores.iloc[max(0, i-29):i+1].mean() + 0.3 * scores.iloc[max(0, i-29):i+1].min(),
        })

data = pd.DataFrame(rows)

for threshold_normal, threshold_degraded in [(90, 75), (85, 65), (80, 60)]:
    def status(h):
        if h >= threshold_normal: return "NORMAL"
        if h >= threshold_degraded: return "DEGRADED"
        return "CRITICAL"
    data["status"] = data["health_score"].apply(status)

    print(f"\n=== Thresholds: NORMAL>={threshold_normal}, DEGRADED>={threshold_degraded} ===")
    print("RUL<10 (should be mostly CRITICAL):")
    print(data[data["true_rul"] < 10]["status"].value_counts(normalize=True).round(3))
    print("RUL>150 (should be mostly NORMAL):")
    print(data[data["true_rul"] > 150]["status"].value_counts(normalize=True).round(3))