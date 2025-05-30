import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import joblib
import os
import matplotlib.pyplot as plt


class AirQualityModel:
    def __init__(self):
        self.pollutants = ['pm2_5', 'pm10', 'no2', 'o3', 'co']
        self.create_folders()

    def create_folders(self):
        for folder in ['data', 'models', 'visualizations']:
            if not os.path.exists(folder):
                os.makedirs(folder)

    def load_and_preprocess_data(self):
        pollutant_dfs = {}
        for name in self.pollutants:
            df = pd.read_csv(f'data/{name}.csv', sep=";")
            df = df[df['code qualité'] == 'A']
            pollutant_dfs[name] = df[['Date de début', 'code site', 'nom site', 'valeur',
                                      'unité de mesure', 'code qualité', 'Latitude', 'Longitude']].rename(
                columns={'valeur': name})

        self.data = pollutant_dfs[self.pollutants[0]]
        for df in list(pollutant_dfs.values())[1:]:
            self.data = self.data.merge(df, how='outer',
                                        on=['Date de début', 'code site', 'nom site',
                                            'unité de mesure', 'code qualité', 'Latitude', 'Longitude'])

        self.data = self.data.rename(columns={
            'Date de début': 'date_start', 'code site': 'site_code',
            'nom site': 'site_name', 'unité de mesure': 'unit_measure',
            'code qualité': 'code_quality', 'Latitude': 'latitude',
            'Longitude': 'longitude'
        })

        self.data['date_start'] = pd.to_datetime(self.data['date_start'])
        temporal_features = pd.DataFrame({
            'jour_semaine': self.data['date_start'].dt.dayofweek,
            'mois': self.data['date_start'].dt.month,
            'saison': self.data['date_start'].dt.month % 12 // 3 + 1,
            'jour_annee': self.data['date_start'].dt.dayofyear
        })

        self.data = pd.concat([self.data, temporal_features], axis=1)

        imputer = SimpleImputer(strategy='mean')
        self.data[self.pollutants] = imputer.fit_transform(self.data[self.pollutants])

        return self.create_features()

    def create_features(self):
        all_features = []
        for pollutant in self.pollutants:
            features_dict = {}

            for lag in [1, 2, 3, 7, 14, 21, 28]:
                features_dict[f'{pollutant}_lag{lag}'] = self.data.groupby('site_code')[pollutant].shift(lag)

            for window in [3, 7, 14, 21, 28]:
                rolling = self.data.groupby('site_code')[pollutant].rolling(window=window, min_periods=1)
                features_dict[f'{pollutant}_rolling{window}'] = rolling.mean().reset_index(0, drop=True)
                features_dict[f'{pollutant}_std{window}'] = rolling.std().reset_index(0, drop=True)
                features_dict[f'{pollutant}_max{window}'] = rolling.max().reset_index(0, drop=True)
                features_dict[f'{pollutant}_min{window}'] = rolling.min().reset_index(0, drop=True)

            for period in [1, 7, 14, 28]:
                features_dict[f'{pollutant}_diff{period}'] = self.data.groupby('site_code')[pollutant].diff(period)

            for span in [7, 14, 28]:
                features_dict[f'{pollutant}_ewm{span}'] = self.data.groupby('site_code')[pollutant].ewm(
                    span=span).mean().reset_index(0, drop=True)

            all_features.append(pd.DataFrame(features_dict))

        self.data = pd.concat([self.data] + all_features, axis=1)
        self.data = self.data.dropna().copy()
        return self.data

    def get_pollutant_features(self, pollutant):
        base_features = ['jour_semaine', 'mois', 'saison', 'jour_annee', 'latitude', 'longitude']

        feature_patterns = [
                               f'{pollutant}_lag{lag}' for lag in [1, 2, 3, 7, 14, 21, 28]
                           ] + [
                               f'{pollutant}_{pattern}{window}'
                               for pattern in ['rolling', 'std', 'max', 'min']
                               for window in [3, 7, 14, 21, 28]
                           ] + [
                               f'{pollutant}_diff{period}' for period in [1, 7, 14, 28]
                           ] + [
                               f'{pollutant}_ewm{span}' for span in [7, 14, 28]
                           ]

        if pollutant == 'pm10':
            feature_patterns.extend([
                'pm2_5_lag1', 'pm2_5_lag7', 'pm2_5_lag14',
                'pm2_5_rolling7', 'pm2_5_rolling14', 'pm2_5_rolling28',
                'pm2_5_std7', 'pm2_5_max7', 'pm2_5_min7',
                'no2_lag1', 'no2_rolling7', 'no2_std7'
            ])

        return base_features + [f for f in feature_patterns if f in self.data.columns]

    def train_models(self):
        self.models = {}
        self.results = {}
        self.predictions = {}

        model_configs = {
            'pm10': (RandomForestRegressor, {
                'n_estimators': [300],
                'max_depth': [15],
                'min_samples_split': [5],
                'min_samples_leaf': [2],
                'max_features': ['sqrt']
            }),
            'co': (XGBRegressor, {
                'n_estimators': [100],
                'learning_rate': [0.05],
                'max_depth': [5],
                'subsample': [0.8]
            }),
            'default': (GradientBoostingRegressor, {
                'n_estimators': [200],
                'learning_rate': [0.1],
                'max_depth': [3],
                'subsample': [0.8]
            })
        }

        for pollutant in self.pollutants:
            print(f"\nTraining model for {pollutant}")

            features = self.get_pollutant_features(pollutant)
            X = self.data[features]
            y = self.data[pollutant]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )

            config_key = pollutant if pollutant in ['pm10', 'co'] else 'default'
            model_class, params = model_configs[config_key]
            model = GridSearchCV(
                model_class(random_state=42),
                params,
                cv=5,
                scoring='r2',
                n_jobs=-1,
                verbose=1
            )

            model.fit(X_train, y_train)
            # Sauvegarder les noms des features utilisées
            joblib.dump(X_train.columns.tolist(), f'models/features_{pollutant}.joblib')

            y_pred = model.predict(X_test)

            self.models[pollutant] = model.best_estimator_
            self.predictions[pollutant] = {'true': y_test, 'pred': y_pred}

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            self.results[pollutant] = {
                'RMSE': rmse,
                'R2': r2,
                'Best_Parameters': model.best_params_,
                'Best_CV_Score': model.best_score_
            }

            print(f"Results for {pollutant}:")
            print(f"Best parameters: {model.best_params_}")
            print(f"RMSE: {rmse:.4f}")
            print(f"R2 Score: {r2:.4f}")
            print(f"Best CV Score: {model.best_score_:.4f}")

        self.save_results()
        self.create_visualizations()

    def create_visualizations(self):
        plt.figure(figsize=(12, 6))
        r2_scores = [self.results[p]['R2'] for p in self.pollutants]
        plt.bar(self.pollutants, r2_scores)
        plt.title('Précision du modèle par polluant (R²)')
        plt.ylabel('R² Score')
        plt.ylim(0, 1)
        for i, v in enumerate(r2_scores):
            plt.text(i, v + 0.01, f'{v:.2%}', ha='center')
        plt.savefig('visualizations/model_performance.png')
        plt.close()

        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.ravel()

        for i, pollutant in enumerate(self.pollutants):
            true_values = self.predictions[pollutant]['true']
            pred_values = self.predictions[pollutant]['pred']

            axs[i].scatter(true_values, pred_values, alpha=0.5)
            axs[i].plot([true_values.min(), true_values.max()],
                        [true_values.min(), true_values.max()],
                        'r--', lw=2)
            axs[i].set_title(f'{pollutant} - R² = {self.results[pollutant]["R2"]:.2%}')
            axs[i].set_xlabel('Valeurs réelles')
            axs[i].set_ylabel('Prédictions')

        plt.tight_layout()
        plt.savefig('visualizations/predictions_vs_reality.png')
        plt.close()

        fig, axs = plt.subplots(len(self.pollutants), 1, figsize=(15, 4 * len(self.pollutants)))

        for i, pollutant in enumerate(self.pollutants):
            true_values = self.predictions[pollutant]['true']
            pred_values = self.predictions[pollutant]['pred']

            axs[i].plot(true_values[-100:].values, label='Réel', alpha=0.7)
            axs[i].plot(pred_values[-100:], label='Prédit', alpha=0.7)
            axs[i].set_title(f'Évolution des valeurs de {pollutant}')
            axs[i].legend()
            axs[i].set_xlabel('Temps')
            axs[i].set_ylabel('Valeur')

        plt.tight_layout()
        plt.savefig('visualizations/temporal_evolution.png')
        plt.close()

    def save_results(self):
        for pollutant, model in self.models.items():
            joblib.dump(model, f'models/model_{pollutant}_final.joblib')

        results_df = pd.DataFrame(self.results).T
        results_df.to_csv('models/final_models_results.csv')

    def predict(self, new_data):
        predictions = {}
        for pollutant in self.pollutants:
            features = self.get_pollutant_features(pollutant)
            available_features = [f for f in features if f in new_data.columns]
            X = new_data[available_features]
            predictions[pollutant] = self.models[pollutant].predict(X)
        return predictions


if __name__ == "__main__":
    model = AirQualityModel()
    model.load_and_preprocess_data()
    model.train_models()
