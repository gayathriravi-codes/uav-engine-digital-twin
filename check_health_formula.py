import pandas as pd
from schema import HEALTHY_RANGES

df = pd.read_csv("data/raw/overheat_000.csv")
window = df.iloc[-30:]

print(window[["egt", "cht", "rpm", "true_rul_timesteps"]].describe())
print()
print("egt healthy range:", HEALTHY_RANGES["egt"])
print("cht healthy range:", HEALTHY_RANGES["cht"])