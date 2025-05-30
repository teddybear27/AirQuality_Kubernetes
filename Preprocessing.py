import pandas as pd
from sklearn.impute import SimpleImputer

# Charger les données
pollutants = {name: pd.read_csv(f'data/{name}.csv', sep=";")[
    lambda df: df['code qualité'] == 'A'
][['Date de début', 'code site', 'nom site', 'valeur', 'unité de mesure',
   'code qualité', 'Latitude', 'Longitude']].rename(columns={'valeur': name})
    for name in ['pm2_5', 'pm10', 'no2', 'o3', 'co']}

# Fusion et renommage
merged_data = pollutants['pm2_5']
for df in list(pollutants.values())[1:]:
    merged_data = merged_data.merge(df, how='outer', on=[
        'Date de début', 'code site', 'nom site', 'unité de mesure',
        'code qualité', 'Latitude', 'Longitude'])

merged_data = merged_data.rename(columns={
    'Date de début': 'date_start', 'code site': 'site_code',
    'nom site': 'site_name', 'unité de mesure': 'unit_measure',
    'code qualité': 'code_quality', 'Latitude': 'latitude',
    'Longitude': 'longitude'})

# Features temporelles
merged_data['date_start'] = pd.to_datetime(merged_data['date_start'])
merged_data['jour_semaine'] = merged_data['date_start'].dt.dayofweek
merged_data['mois'] = merged_data['date_start'].dt.month
merged_data['saison'] = merged_data['date_start'].dt.month % 12 // 3 + 1

# Imputation
pollutants = ['pm2_5', 'pm10', 'no2', 'o3', 'co']
imputer = SimpleImputer(strategy='mean')
merged_data[pollutants] = imputer.fit_transform(merged_data[pollutants])

# Stats et sauvegarde
print(f"Observations totales: {len(merged_data)}")
print(f"Sites: {merged_data['site_code'].nunique()}")
print("\nObservations par site:")
print(merged_data.groupby('site_code').size())

merged_data.to_csv('data/final_air_quality_data.csv', index=False)