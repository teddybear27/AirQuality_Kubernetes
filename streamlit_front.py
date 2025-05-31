import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import folium
from streamlit_folium import folium_static
import datetime
from sklearn.preprocessing import LabelEncoder

# URLs des collaborateurs
url1 = "https://www.linkedin.com/in/rimey-aboky-25603a20b/"
url2 = "https://www.linkedin.com/in/nene-aminatou-diallo/"
url3 = "https://www.linkedin.com/in/cheikhkane0104/"
url4 = "https://fr.linkedin.com/in/afoezon"

# Chargement des modèles
polluants_models = {
    'pm2_5': joblib.load("models/model_pm2_5_final.joblib"),
    'pm10': joblib.load("models/model_pm10_final.joblib"),
    'no2': joblib.load("models/model_no2_final.joblib"),
    'o3': joblib.load("models/model_o3_final.joblib"),
    'co': joblib.load("models/model_co_final.joblib")
}

# Charger les features pour chaque modèle
expected_features = {
    pollutant: joblib.load(f"models/features_{pollutant}.joblib")
    for pollutant in polluants_models.keys()
}

# Chargement des données
df = pd.read_csv("data/final_air_quality_data.csv")

label_encoder_localite = LabelEncoder()
df['site_name'] = label_encoder_localite.fit_transform(df['site_name'])

label_encoder_site_code = LabelEncoder()
df['site_code'] = label_encoder_site_code.fit_transform(df['site_code'])

# Interface Streamlit
st.title("ENV-AI : La startup qui utilise l'intelligence Artificielle au service de l'Environnement")
st.sidebar.write("ENV-AI Collaborateurs")

st.sidebar.write("[Rimey ABOKY](%s)" % url1)
st.sidebar.write("[Aminatou DIALLO](%s)" % url2)
st.sidebar.write("[Cheikh KANE](%s)" % url3)
st.sidebar.write("[Adrian FOEZON](%s)" % url4)

st.sidebar.markdown(
    "**Comment fonctionne ENV-AI** \n"
    "1. Sélectionnez une date\n"
    "2. Sélectionnez une zone de Paris\n"
    "3. Prenez vos précautions\n"
)

st.sidebar.header("Options utilisateur")

# Sélection de la localité
site_names = df['site_name'].unique()
site_name_label = st.sidebar.selectbox("Choisissez une localité :",
                                       label_encoder_localite.inverse_transform(site_names))
site_name = label_encoder_localite.transform([site_name_label])[0]

# Sélection de la date
selected_date = st.sidebar.date_input(
    "Choisissez une date :",
    datetime.date(2024, 12, 20),
    min_value=datetime.date(2024, 12, 1),
    max_value=datetime.date(2030, 12, 31)
)

# Jour de la semaine
jour = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'][selected_date.weekday()]

# Carte interactive
st.header(f"Carte de la localité : {site_name_label}")
site_name_data = df[df['site_name'] == site_name].iloc[0]
latitude, longitude = site_name_data['latitude'], site_name_data['longitude']

m = folium.Map(location=[latitude, longitude], zoom_start=13)
folium.Marker([latitude, longitude], popup=f"Localité : {site_name_label}").add_to(m)
folium_static(m)

# Section prédictions
st.header(f"Prédictions pour la localité : {site_name_label} ({jour})")


# Fonction pour synchroniser les features
def synchronize_features(input_data, expected_features):
    # Ajouter les colonnes manquantes avec une valeur par défaut
    for feature in expected_features:
        if feature not in input_data.columns:
            input_data[feature] = 0
    return input_data[expected_features]


# Fonction pour reproduire les transformations des features
def prepare_features(date, site_data, historical_data, expected_features):
    features = {
        'jour_semaine': date.weekday(),
        'mois': date.month,
        'saison': (date.month % 12 // 3) + 1,
        'jour_annee': date.timetuple().tm_yday,
        'latitude': site_data['latitude'],
        'longitude': site_data['longitude']
    }

    for pollutant in ['pm2_5', 'pm10', 'no2', 'o3', 'co']:
        for lag in [1, 7, 14, 28]:
            features[f'{pollutant}_lag{lag}'] = historical_data[pollutant].shift(lag).iloc[-1]
        for window in [3, 7, 14, 28]:
            rolling = historical_data[pollutant].rolling(window=window, min_periods=1)
            features[f'{pollutant}_rolling{window}'] = rolling.mean().iloc[-1]
            features[f'{pollutant}_std{window}'] = rolling.std().iloc[-1]
            features[f'{pollutant}_max{window}'] = rolling.max().iloc[-1]
            features[f'{pollutant}_min{window}'] = rolling.min().iloc[-1]
        for period in [1, 7, 14, 28]:
            features[f'{pollutant}_diff{period}'] = historical_data[pollutant].diff(period).iloc[-1]
        for span in [7, 14, 28]:
            features[f'{pollutant}_ewm{span}'] = historical_data[pollutant].ewm(span=span).mean().iloc[-1]

    input_data = pd.DataFrame([features])

    # Synchroniser les features avec celles attendues
    return synchronize_features(input_data, expected_features)


# Seuils recommandés pour chaque polluant
seuils_polluants = {
    'pm2_5': 10,  # µg/m³
    'pm10': 50,  # µg/m³
    'no2': 40,  # µg/m³
    'o3': 100,  # µg/m³
    'co': 10  # mg/m³
}

predictions = {}
voir = st.checkbox("Afficher les prédictions et les recommandations")

if voir:
    historical_data = df[df['site_name'] == site_name].sort_values(by='date_start')

    # Générer les features et effectuer les prédictions
    for pollutant, model in polluants_models.items():
        input_data = prepare_features(selected_date, site_name_data, historical_data, expected_features[pollutant])
        prediction = model.predict(input_data)[0]
        predictions[pollutant] = prediction

        st.write(f"Polluant {pollutant} : {prediction:.2f} µg/m³")

        seuil = seuils_polluants[pollutant]
        if prediction > seuil:
            st.warning(
                f"La concentration de **{pollutant.upper()}** dépasse le seuil recommandé ({seuil} µg/m³). "
                f"Env-AI vous conseille de limiter les activités en extérieur dans cette zone si vous êtes fragiles du poumon."
            )
        else:
            st.success(
                f"La concentration de **{pollutant.upper()}** est inférieure au seuil recommandé ({seuil} µg/m³). "
                f"La qualité de l'air est acceptable pour les activités extérieures."
            )

    # Visualisation graphique
    couleurs = ['blue', 'red', 'green', 'purple', 'brown']
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(predictions.keys(), predictions.values(), label="Prédictions", color='lightblue')

    for i, (pollutant, seuil) in enumerate(seuils_polluants.items()):
        ax.axhline(y=seuil, color=couleurs[i], linestyle='--', label=f'Seuil {pollutant.upper()}')

    ax.set_xlabel('Polluants')
    ax.set_ylabel('Concentration (µg/m³)')
    ax.set_title('Prédictions de la concentration des polluants par rapport aux seuils recommandés')
    ax.legend()

    st.pyplot(fig)
