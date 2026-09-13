import sys
sys.path.insert(0, 'models/rul')
import numpy as np
from train_rul_ensemble import load_all_flights, build_windowed_dataset_v5

flights = load_all_flights()
X, y, flight_ids = build_windowed_dataset_v5(flights)

print(f'Total windows: {len(y)}')
print(f'Windows with true RUL >= 250: {(y>=250).sum()} ({100*(y>=250).sum()/len(y):.1f}%)')
print(f'Windows with true RUL >= 200: {(y>=200).sum()} ({100*(y>=200).sum()/len(y):.1f}%)')
print(f'Windows with true RUL >= 150: {(y>=150).sum()} ({100*(y>=150).sum()/len(y):.1f}%)')
print(f'Windows with true RUL < 50: {(y<50).sum()} ({100*(y<50).sum()/len(y):.1f}%)')