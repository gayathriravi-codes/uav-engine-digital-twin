import pandas as pd

a = pd.read_csv('data/raw/cooling_degradation_000.csv')
b = pd.read_csv('data/raw/misfire_000.csv')

cols = ['rpm', 'egt', 'cht', 'oil_pressure', 'oil_temp', 'vibration', 'fuel_flow']

print('First 10 rows identical?', a[cols].head(10).equals(b[cols].head(10)))
print()
print("cooling_degradation_000:")
print(a[cols].head(5))
print()
print("misfire_000:")
print(b[cols].head(5))