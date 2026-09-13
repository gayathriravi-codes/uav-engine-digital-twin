import torch
import joblib
import numpy as np
import glob
import pandas as pd
from models.rul.train_rul import RULRegressor
from schema import SENSOR_FIELDS, WINDOW_SIZE

model = RULRegressor()
model.load_state_dict(torch.load("models/rul/rul_model.pt"))
model.eval()
scaler = joblib.load("models/rul/rul_scaler.joblib")

df = pd.read_csv(sorted(glob.glob("data/raw/*.csv"))[0]).sort_values("timestamp").reset_index(drop=True)
print("flight length:", len(df), "| true_rul at last row:", df["true_rul_timesteps"].iloc[-1])

window = df[SENSOR_FIELDS].to_numpy(dtype=np.float32)[-WINDOW_SIZE:]
scaled = scaler.transform(window).reshape(1, WINDOW_SIZE, -1)
with torch.no_grad():
    raw = model(torch.tensor(scaled, dtype=torch.float32)).item()
print("raw (unclamped) prediction:", raw)