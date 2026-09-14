# models/rul/diagnose_250_350_bucket.py
import numpy as np
from train_rul_ensemble import load_ensemble, load_all_flights, build_windowed_dataset_v5
from calibrate_rul import _bucket_index, BUCKET_EDGES
from train_rul_ensemble import predict_rul_ensemble
from sklearn.model_selection import train_test_split

models, scaler, dropped_idx_list = load_ensemble()

flights = load_all_flights()
X, y, flight_ids = build_windowed_dataset_v5(flights)

unique_flights = np.unique(flight_ids)
train_flights, temp_flights = train_test_split(unique_flights, test_size=0.3, random_state=42)
val_flights, test_flights = train_test_split(temp_flights, test_size=0.5, random_state=42)
val_mask = np.isin(flight_ids, val_flights)
X_val, y_val = X[val_mask], y[val_mask]

# Look only at windows whose TRUE RUL is in the (250,350) bucket
target_mask = (y_val >= 250) & (y_val < 350)
print(f"True windows in (250,350): {target_mask.sum()}")

raw_preds = np.array([
    predict_rul_ensemble(X_val[i], models, scaler, dropped_idx_list, calibrated=False)["point_estimate_timesteps"]
    for i in np.where(target_mask)[0]
])

print(f"Raw predictions for these windows: min={raw_preds.min():.1f}  max={raw_preds.max():.1f}  mean={raw_preds.mean():.1f}  std={raw_preds.std():.1f}")
print(f"How many of THOSE raw predictions actually land in (250,350)? {((raw_preds>=250)&(raw_preds<350)).sum()} / {len(raw_preds)}")