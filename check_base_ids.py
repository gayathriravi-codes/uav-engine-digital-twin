import pandas as pd
df = pd.read_csv('data/processed/classification_features.csv')
df['base_id'] = df['flight_id'].str.rsplit('_', n=1).str[-1]
print(df.groupby('base_id')['flight_id'].unique().apply(list).to_string())