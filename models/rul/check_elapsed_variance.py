import sys
sys.path.insert(0, 'models/rul')
import numpy as np
from train_rul_ensemble import load_all_flights
from schema import SENSOR_FIELDS

flights = load_all_flights()
elapsed_fracs_250_350 = []

for flight_id, df in flights:
    rul = df["true_rul_timesteps"].values
    flight_len = len(df)
    window_size = 30
    stride = 5
    for start in range(0, len(df) - window_size + 1, stride):
        end = start + window_size
        true_rul = rul[end - 1]
        if 250 <= true_rul < 350:
            elapsed_fracs_250_350.append(end / flight_len)

elapsed_fracs_250_350 = np.array(elapsed_fracs_250_350)
print(f"n={len(elapsed_fracs_250_350)}")
print(f"elapsed_fraction for RUL>=250 windows: min={elapsed_fracs_250_350.min():.3f}  "
      f"max={elapsed_fracs_250_350.max():.3f}  mean={elapsed_fracs_250_350.mean():.3f}  "
      f"std={elapsed_fracs_250_350.std():.4f}")