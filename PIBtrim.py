import streamlit as st
import pandas as pd
import numpy as np
import random
import re
import unicodedata
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.pipeline import Pipeline
from datetime import datetime
import pytz
import plotly.express as px
import plotly.graph_objects as go
import os
import shap
import matplotlib.pyplot as plt
import joblib
import hashlib
import csv
import time
from io import BytesIO
import logging

# Configuration du journal de logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration initiale
st.title("🔍 Prédiction du PIB trimestriel pour 2024")
cet = pytz.timezone('CET')
current_date_time = cet.localize(datetime(2025, 7, 26, 21, 26))
st.write(f"**Date et heure actuelles :** {current_date_time.strftime('%d/%m/%Y %H:%M %Z')}")

random.seed(42)
np.random.seed(42)

# Initialisation du journal des erreurs
error_log = []

# Fonction de normalisation des chaînes
def normalize_name(name):
    if pd.isna(name) or not isinstance(name, str):
        error_log.append(f"Valeur non textuelle ou NaN : {name}. Remplacement par 'inconnu'.")
        return "inconnu"
    name = re.sub(r'\s+', ' ', name.strip())
    name = name.replace("_", " ")
    name = name.replace("lahabillement", "l'habillement")
    name = name.replace("intarieur", "interieur")
    name = name.replace("activitas", "activites")
    name = name.replace("impa ts", "impots")
    name = name.replace("d'autre produits", "d'autres produits")
    name = name.replace("taux d'interest", "taux d'interet")
    name = name.replace("police monetaire internationale", "politique monetaire internationale")
    name = name.replace("crises sociales", "crise sociale")
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii').strip()
    name = re.sub(r'\s+', ' ', name).lower()
    return name

# Conversion des trimestres en chiffres romains
def convert_roman_to_quarter(index_str):
    roman_to_arabic = {'i': '1', 'ii': '2', 'iii': '3', 'iv': '4'}
    match = re.match(r'^(I|II|III|IV)\s*trimestre\s*(\d{4})$', index_str, re.IGNORECASE)
    if not match:
        error_log.append(f"Format de trimestre invalide : {index_str}")
        raise ValueError(f"Format de trimestre invalide : {index_str}")
    roman_quarter, year = match.groups()
    quarter = roman_to_arabic[roman_quarter.lower()]
    return f"{year}Q{quarter}"

# Calcul du hash du fichier pour validation du cache
def compute_file_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

# Chargement et prétraitement des données
@st.cache_data(show_spinner=False, persist=True)
def load_and_preprocess(uploaded_file=None, _cache_key="default"):
    start_time = time.time()
    try:
        if uploaded_file:
            uploaded_file.seek(0)
            raw_content = uploaded_file.read()
            if not raw_content.strip():
                error_log.append("Le fichier uploadé est vide.")
                st.error("Erreur : Le fichier uploadé est vide. Veuillez vérifier le fichier.")
                raise ValueError("Fichier vide.")
            error_log.append(f"Contenu brut du fichier uploadé (premiers 200 octets) : {raw_content[:200].decode('utf-8', errors='replace')}...")
            
            encodings = ['utf-8', 'latin-1', 'windows-1252', 'iso-8859-1']
            separators = [';', ',', '\t', '|', ':']
            df = None
            for encoding in encodings:
                for sep in separators:
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(
                            uploaded_file,
                            thousands=' ', 
                            decimal=',',
                            encoding=encoding,
                            sep=sep,
                            skipinitialspace=True,
                            engine='c',
                            dtype_backend='numpy_nullable'
                        )
                        if not df.empty and len(df.columns) > 1:
                            first_col = df.columns[0].lower().strip()
                            if first_col.startswith('sect') or first_col in ['\ufeffsecteur', 'sector']:
                                error_log.append(f"Fichier chargé avec encodage '{encoding}' et séparateur '{sep}'.")
                                break
                    except Exception as e:
                        error_log.append(f"Échec de lecture avec encodage '{encoding}' et séparateur '{sep}': {str(e)}")
                if df is not None:
                    break
            else:
                st.error("Échec de la lecture automatique du CSV. Veuillez spécifier l'encodage et le séparateur.")
                encoding = st.selectbox("Choisir l'encodage", encodings)
                sep = st.text_input("Entrer le séparateur (e.g., ';', ',', '\\t')", value=';')
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(
                        uploaded_file,
                        thousands=' ', 
                        decimal=',',
                        encoding=encoding,
                        sep=sep,
                        skipinitialspace=True,
                        engine='c'
                    )
                    error_log.append(f"Fichier chargé avec encodage manuel '{encoding}' et séparateur '{sep}'.")
                except Exception as e:
                    error_log.append(f"Échec de lecture avec encodage manuel '{encoding}' et séparateur '{sep}': {str(e)}")
                    st.error(f"Erreur : Impossible de lire le fichier CSV avec les paramètres fournis : {str(e)}")
                    raise ValueError("Format CSV invalide ou paramètres incorrects.")

            if df is None:
                error_log.append("Aucun DataFrame valide n'a pu être chargé.")
                st.error("Erreur : Impossible de lire le fichier CSV. Vérifiez le contenu du fichier.")
                raise ValueError("Aucun DataFrame valide chargé.")
        else:
            default_file = "PIB_Trimestrielle.csv"
            if not os.path.exists(default_file):
                error_log.append(f"Fichier '{default_file}' introuvable.")
                st.error(f"Erreur : Fichier '{default_file}' introuvable. Vérifiez le chemin du fichier.")
                raise FileNotFoundError(f"Fichier '{default_file}' introuvable.")
            df = pd.read_csv(
                default_file,
                thousands=' ',
                decimal=',',
                encoding='latin-1',
                sep=';',
                engine='c',
                dtype_backend='numpy_nullable'
            )
            error_log.append(f"Fichier chargé comme CSV avec encodage 'latin-1' et séparateur ';'.")

        if df.empty or len(df.columns) == 0:
            error_log.append("Le fichier CSV ne contient aucune colonne valide.")
            st.error("Erreur : Le fichier CSV ne contient aucune colonne valide.")
            raise ValueError("Aucune colonne dans le fichier CSV.")
        
        first_col = df.columns[0].lower().strip()
        if not (first_col.startswith('sect') or first_col in ['\ufeffsecteur', 'sector']):
            df.columns = ['Secteur'] + list(df.columns[1:])
            error_log.append(f"Première colonne renommée en 'Secteur' (précédemment : '{first_col}')")
        
        if 'Secteur' not in df.columns:
            st.error(f"Erreur : La colonne 'Secteur' n'est pas présente. Colonnes actuelles : {list(df.columns)}")
            error_log.append(f"Colonne 'Secteur' absente. Colonnes trouvées : {df.columns.tolist()}")
            raise KeyError("Colonne 'Secteur' manquante après renommage.")

        df['Secteur'] = df['Secteur'].apply(normalize_name)
        df.columns = ['Secteur' if col == 'Secteur' else normalize_name(col) for col in df.columns]
        for col in df.columns[1:]:
            df[col] = df[col].astype(str).str.replace(' ', '').str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('float32')

        macro_keywords = [
            "taux de chomage", "taux d'inflation", "taux d'interet", "dette publique",
            "politique monetaire internationale", "tensions geopolitiques regionales",
            "prix matieres premieres", "secheresse et desastre climatique", "pandemies",
            "crise sociale"
        ]
        sectors = [
            "agriculture, sylviculture et peche", "extraction petrole et gaz naturel",
            "extraction des produits miniers", "industries agro-alimentaires",
            "industrie du textile, de l'habillement et du cuir",
            "raffinage du petrole", "chimiques", "materiaux de construction, ceramique et verre",
            "mecaniques et electriques", "industries diverses",
            "production et distribution de l'electricite et du gaz",
            "distribution d'eau et traitement des eaux usees et des dechets",
            "construction", "commerce, entretien et reparation",
            "services du transport et entreposage",
            "secteur des services d'hotellerie, de cafe et de restauration",
            "information et communication", "activites financieres",
            "services d'administration publique et de defense",
            "enseignement prive et publique", "sante et action sociale prive et publique",
            "autres services marchands", "autres activites des menages",
            "services fournis par les organisations associatives",
            "activites marchandes", "activites non marchandes",
            "impots nets de subventions sur les produits"
        ]
        macro_rates = [
            "taux de chomage", "taux d'inflation", "taux d'interet", "dette publique"
        ]
        events = [
            "politique monetaire internationale", "tensions geopolitiques regionales",
            "prix matieres premieres", "secheresse et desastre climatique", "pandemies",
            "crise sociale"
        ]

        macro_keywords = [normalize_name(m) for m in macro_keywords]
        sectors = [normalize_name(s) for s in sectors]
        macro_rates = [normalize_name(m) for m in macro_rates]
        events = [normalize_name(e) for e in events]

        actual_sectors = df['Secteur'].tolist()
        error_log.append(f"Secteurs dans le CSV : {actual_sectors}")

        df_macro = df[df['Secteur'].isin(macro_keywords)].copy()
        df_pib = df[df['Secteur'] == "produit interieur brut"].copy()
        if df_pib.empty:
            possible_gdp_names = ['produit intarieur brut', 'produit intérieur brut', 'Produit Intérieur Brut']
            for gdp_name in possible_gdp_names:
                if normalize_name(gdp_name) in df['Secteur'].values:
                    df_pib = df[df['Secteur'] == normalize_name(gdp_name)].copy()
                    df_pib['Secteur'] = 'produit interieur brut'
                    break
            if df_pib.empty:
                st.error("Erreur : Aucune donnée PIB trouvée même après recherche de variantes. Colonnes disponibles : {}".format(df['Secteur'].tolist()))
                error_log.append("Aucune donnée PIB trouvée dans le fichier.")
                raise ValueError("Données PIB manquantes dans le CSV.")

        df_secteurs = df[df['Secteur'].isin(sectors) & ~df['Secteur'].str.contains("PIB", case=False)].copy()

        missing_sectors = [s for s in sectors if s not in df['Secteur'].values]
        missing_macro = [m for m in macro_keywords if m not in df['Secteur'].values]
        if missing_sectors:
            st.warning(f"Attention : Secteurs manquants dans le CSV : {missing_sectors}. Utilisation de la moyenne des secteurs disponibles.")
            error_log.append(f"Secteurs manquants dans le CSV : {missing_sectors}")
        if missing_macro:
            st.warning(f"Attention : Macros manquants : {missing_macro}. Utilisation de valeurs par défaut (0).")
            error_log.append(f"Macros manquants : {missing_macro}")

        df_macro.set_index("Secteur", inplace=True)
        df_pib.set_index("Secteur", inplace=True)
        df_secteurs.set_index("Secteur", inplace=True)

        df_macro_T = df_macro.transpose()
        df_pib_T = df_pib.transpose()
        df_secteurs_T = df_secteurs.transpose()

        try:
            df_macro_T.index = [convert_roman_to_quarter(idx) for idx in df_macro_T.index]
            df_pib_T.index = [convert_roman_to_quarter(idx) for idx in df_pib_T.index]
            df_secteurs_T.index = [convert_roman_to_quarter(idx) for idx in df_secteurs_T.index]
            df_macro_T.index = pd.PeriodIndex(df_macro_T.index, freq='Q').to_timestamp()
            df_pib_T.index = pd.PeriodIndex(df_pib_T.index, freq='Q').to_timestamp()
            df_secteurs_T.index = pd.PeriodIndex(df_secteurs_T.index, freq='Q').to_timestamp()
        except ValueError as e:
            st.error(f"Erreur lors de la conversion des indices en dates : {str(e)}")
            error_log.append(f"Erreur lors de la conversion des indices : {str(e)}")
            raise

        X_df = pd.concat([df_secteurs_T, df_macro_T], axis=1).dropna()
        error_log.append(f"Forme de X_df après concaténation : {X_df.shape}")
        error_log.append(f"Colonnes de X_df après concaténation : {list(X_df.columns)}")

        y_df = df_pib_T.loc[X_df.index]
        if y_df.empty:
            st.error("Erreur : y_df vide après alignement avec X_df. Indices X_df : {}. Indices df_pib_T : {}".format(X_df.index.tolist(), df_pib_T.index.tolist()))
            error_log.append("y_df vide après alignement.")
            raise ValueError("Données PIB vides après prétraitement.")

        key_sectors = [
            "agriculture, sylviculture et peche",
            "mecaniques et electriques",
            "secteur des services d'hotellerie, de cafe et de restauration",
            "information et communication",
            "activites financieres"
        ]
        key_sectors = [normalize_name(s) for s in key_sectors]

        for sector in key_sectors:
            col_name = f"{sector}_lag1"
            if sector in X_df.columns:
                X_df[col_name] = X_df[sector].shift(1).fillna(X_df[sector].mean())
            else:
                X_df[col_name] = X_df[sectors].mean(axis=1).shift(1).fillna(X_df[sectors].mean().mean()) if sectors else 0
                error_log.append(f"Feature décalée '{col_name}' ajoutée avec moyenne des secteurs car '{sector}' est absent.")

        for rate in macro_rates:
            col_name = f"{rate}_lag1"
            if rate in X_df.columns:
                X_df[col_name] = X_df[rate].shift(1).fillna(X_df[rate].mean())
            else:
                X_df[col_name] = 0
                error_log.append(f"Feature décalée '{col_name}' ajoutée avec valeur 0 car '{rate}' est absent.")

        X_df['gdp_lag1'] = y_df.shift(1).fillna(y_df.mean())

        expected_features = (
            sectors +
            macro_rates +
            events +
            [f"{s}_lag1" for s in key_sectors] +
            [f"{r}_lag1" for r in macro_rates] +
            ['gdp_lag1']
        )
        error_log.append(f"Colonnes attendues dans expected_features : {expected_features} (nombre: {len(expected_features)})")

        missing_cols = [col for col in expected_features if col not in X_df.columns]
        extra_cols = [col for col in X_df.columns if col not in expected_features]
        if missing_cols:
            existing_cols = [col for col in sectors + macro_rates + events if col in X_df.columns]
            for col in missing_cols:
                if col in sectors and existing_cols:
                    X_df[col] = X_df[existing_cols].mean(axis=1)
                    error_log.append(f"Feature manquante '{col}' ajoutée avec la moyenne des secteurs disponibles.")
                elif col.endswith('_lag1') and col.replace('_lag1', '') in X_df.columns:
                    X_df[col] = X_df[col.replace('_lag1', '')].shift(1).fillna(X_df[col.replace('_lag1', '')].mean())
                    error_log.append(f"Feature manquante '{col}' ajoutée avec décalage.")
                else:
                    X_df[col] = 0
                    error_log.append(f"Feature manquante '{col}' ajoutée avec valeur 0.")
        if extra_cols:
            st.warning(f"Attention : Colonnes supplémentaires dans X_df : {extra_cols}")
            error_log.append(f"Colonnes supplémentaires dans X_df : {extra_cols}")
            X_df = X_df.drop(columns=extra_cols, errors='ignore')

        X_df = X_df[expected_features]
        error_log.append(f"Colonnes dans X_df après réordonnancement : {list(X_df.columns)}")
        error_log.append(f"Nombre de colonnes dans X_df : {X_df.shape[1]} (attendu : {len(expected_features)})")

        if list(X_df.columns) != expected_features:
            differences = [(i, a, b) for i, (a, b) in enumerate(zip(X_df.columns, expected_features)) if a != b]
            error_log.append(f"Non-concordance dans les colonnes de X_df : Différences aux positions {differences}")
            st.error(f"Erreur : Les colonnes de X_df ({len(X_df.columns)}) ne correspondent pas à expected_features ({len(expected_features)}). Différences : {differences}")
            st.stop()

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X = scaler_X.fit_transform(X_df)
        y = scaler_y.fit_transform(y_df.values.reshape(-1, 1)).flatten()
        quarters = X_df.index

        last_year = int(max(quarters).year)
        logger.info(f"Prétraitement terminé en {time.time() - start_time:.2f} secondes")
        return X, y, quarters, X_df, scaler_X, scaler_y, macro_keywords, sectors, macro_rates, events, last_year, y_df, expected_features, df

    except Exception as e:
        error_log.append(f"Erreur lors du chargement du fichier : {str(e)}")
        st.error(f"Erreur : Erreur lors du chargement du fichier : {str(e)}")
        raise

# Téléversement du fichier
uploaded_file = st.file_uploader("Téléchargez votre jeu de données mis à jour (CSV, optionnel)", type=["csv"])
if uploaded_file:
    st.cache_data.clear()
    st.cache_resource.clear()
    st.write("### Aperçu du fichier CSV chargé")
    try:
        uploaded_file.seek(0)
        try:
            df_preview = pd.read_csv(uploaded_file, thousands=' ', decimal=',', encoding='utf-8', sep=';')
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df_preview = pd.read_csv(uploaded_file, thousands=' ', decimal=',', encoding='latin-1', sep=';')
        st.write(df_preview)
        
        if st.button("Ajouter une nouvelle ligne"):
            quarter_columns = [col for col in df_preview.columns if re.match(r'^(I|II|III|IV)\s*trimestre\s*\d{4}$', col, re.IGNORECASE)]
            if quarter_columns:
                valid_quarters = []
                for col in quarter_columns:
                    try:
                        period = pd.Period(convert_roman_to_quarter(col), freq='Q')
                        valid_quarters.append((col, period))
                    except ValueError as e:
                        error_log.append(f"Colonne ignorée '{col}' : format de trimestre invalide ({str(e)})")
                        continue
                if valid_quarters:
                    last_quarter, last_period = max(valid_quarters, key=lambda x: x[1])
                    new_period = last_period + 1
                    new_quarter = f"{['I', 'II', 'III', 'IV'][new_period.quarter-1]} trimestre {new_period.year}"
                else:
                    error_log.append("Aucune colonne de trimestre valide trouvée. Utilisation de 'I trimestre 2024' par défaut.")
                    new_quarter = "I trimestre 2024"
            else:
                error_log.append("Aucune colonne de trimestre détectée. Utilisation de 'I trimestre 2024' par défaut.")
                new_quarter = "I trimestre 2024"
            new_row = pd.DataFrame({col: ['produit interieur brut' if col == 'Secteur' else 0.0] for col in df_preview.columns})
            if new_quarter not in df_preview.columns:
                new_row[new_quarter] = 0.0
            st.write(f"### Ajouter des données pour {new_quarter}")
            edited_row = st.data_editor(new_row, num_rows="dynamic")
            
            if st.button("Enregistrer la nouvelle ligne"):
                for col in df_preview.columns:
                    if col not in edited_row.columns:
                        edited_row[col] = 0.0
                if new_quarter not in df_preview.columns:
                    df_preview[new_quarter] = 0.0
                df_updated = pd.concat([df_preview, edited_row], ignore_index=True)
                output_file = "updated_PIB_Trimestrielle.csv"
                df_updated.to_csv(output_file, sep=';', index=False, encoding='latin-1')
                st.success(f"Succès : Nouvelle ligne enregistrée dans '{output_file}'.")
                csv_buffer = BytesIO()
                df_updated.to_csv(csv_buffer, sep=';', index=False, encoding='latin-1')
                csv_buffer.seek(0)
                uploaded_file = csv_buffer
                uploaded_file.name = output_file
    except Exception as e:
        error_log.append(f"Erreur lors de la lecture du fichier uploadé pour l'aperçu : {str(e)}")
        st.error(f"Erreur : Erreur lors de la lecture du fichier uploadé : {str(e)}")
        st.stop()

# Chargement des données
try:
    cache_key = "default" if uploaded_file is None else hashlib.sha256(uploaded_file.read()).hexdigest()
    if uploaded_file:
        uploaded_file.seek(0)
    X, y, quarters, X_df, scaler_X, scaler_y, macro_keywords, sectors, macro_rates, events, last_year, y_df, expected_features, df = load_and_preprocess(uploaded_file, cache_key)
except (ValueError, FileNotFoundError, KeyError) as e:
    st.error(f"Erreur : {str(e)}")
    st.stop()

st.write(f"**Dernière année disponible dans les données :** {last_year}")
st.write(f"**Nombre de features dans X_df :** {X_df.shape[1]} (attendu : {len(expected_features)})")

# Structure du modèle en cache
@st.cache_resource(show_spinner=False)
def get_model_structure(model_type, _cache_key="default"):
    if model_type == "Ridge":
        return Pipeline([
            ('feature_selection', SelectKBest(score_func=mutual_info_regression)),
            ('ridge', Ridge())
        ])
    elif model_type == "ElasticNet":
        return Pipeline([
            ('feature_selection', SelectKBest(score_func=mutual_info_regression)),
            ('elasticnet', ElasticNet())
        ])
    elif model_type == "Huber":
        return Pipeline([
            ('feature_selection', SelectKBest(score_func=mutual_info_regression)),
            ('huber', HuberRegressor(max_iter=1000))
        ])

# Définition des modèles
tscv = TimeSeriesSplit(n_splits=8)
ridge_params = {
    'ridge__alpha': np.logspace(-2, 3, 50),
    'feature_selection__k': [5, 10, 15, 20, 25]
}
ridge_cv = RandomizedSearchCV(
    get_model_structure("Ridge"),
    ridge_params,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_iter=20,
    random_state=42,
    n_jobs=-1
)

elasticnet_params = {
    'elasticnet__alpha': np.logspace(-2, 3, 50),
    'elasticnet__l1_ratio': np.linspace(0.1, 0.9, 9),
    'feature_selection__k': [5, 10, 15, 20, 25]
}
elasticnet_cv = RandomizedSearchCV(
    get_model_structure("ElasticNet"),
    elasticnet_params,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_iter=20,
    random_state=42,
    n_jobs=-1
)

huber_params = {
    'huber__epsilon': np.linspace(1.1, 2.0, 10),
    'huber__alpha': np.logspace(-4, 1, 20),
    'feature_selection__k': [5, 10, 15, 20, 25]
}
huber_cv = RandomizedSearchCV(
    get_model_structure("Huber"),
    huber_params,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_iter=20,
    random_state=42,
    n_jobs=-1
)

# Chargement ou entraînement des modèles
def load_or_train_models(X, y, cache_key):
    default_file = "PIB_Trimestrielle.csv"
    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)
    model_files = {
        "Ridge": os.path.join(model_dir, f"ridge_model_{cache_key}.joblib"),
        "ElasticNet": os.path.join(model_dir, f"elasticnet_model_{cache_key}.joblib"),
        "Huber": os.path.join(model_dir, f"huber_model_{cache_key}.joblib")
    }
    results = []
    models = {}
    test_maes = {}

    if uploaded_file is None and os.path.exists(default_file):
        file_hash = compute_file_hash(default_file)
        if cache_key == file_hash:
            for name, file_path in model_files.items():
                if os.path.exists(file_path):
                    try:
                        model_cv = joblib.load(file_path)
                        train_pred = model_cv.predict(X)
                        train_pred_unscaled = scaler_y.inverse_transform(train_pred.reshape(-1, 1)).flatten()
                        y_unscaled = scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()
                        train_mae = mean_absolute_error(y_unscaled, train_pred_unscaled)
                        train_r2 = r2_score(y_unscaled, train_pred_unscaled)

                        preds_test = []
                        for tr, te in tscv.split(X):
                            best_model = model_cv.best_estimator_
                            best_model.fit(X[tr], y[tr])
                            preds_test.extend(best_model.predict(X[te]))

                        test_pred_unscaled = scaler_y.inverse_transform(np.array(preds_test).reshape(-1, 1)).flatten()
                        test_mae = mean_absolute_error(y_unscaled[-len(preds_test):], test_pred_unscaled)
                        test_r2 = r2_score(y_unscaled[-len(preds_test):], test_pred_unscaled)

                        st.markdown(f"### 🔍 Résultats pour **{name}** (chargé depuis le disque)")
                        st.write(f"MAE d'entraînement : {train_mae:.2f}, MAE de test (TimeSeriesSplit) : {test_mae:.2f}")
                        st.write(f"R² d'entraînement : {train_r2:.4f}, R² de test : {test_r2:.4f}")
                        st.write(f"Meilleurs hyperparamètres : {model_cv.best_params_}")

                        interpret_results(name, train_mae, test_mae, train_r2, test_r2)
                        results.append({
                            'Modèle': name,
                            'CV MAE': test_mae,
                            'Train R²': train_r2
                        })
                        models[name] = model_cv
                        test_maes[name] = test_mae
                        error_log.append(f"Modèle {name} chargé depuis {file_path}")
                    except Exception as e:
                        error_log.append(f"Échec du chargement du modèle {name} depuis {file_path}: {str(e)}")
                        st.warning(f"Attention : Échec du chargement du modèle {name}. Réentraînement...")
                        mae, r2, trained_model = eval_and_detect(globals()[f"{name.lower()}_cv"], X, y, name)
                        joblib.dump(trained_model, file_path)
                        results.append({
                            'Modèle': name,
                            'CV MAE': mae,
                            'Train R²': r2_score(scaler_y.inverse_transform(y.reshape(-1, 1)), scaler_y.inverse_transform(trained_model.predict(X).reshape(-1, 1)))
                        })
                        models[name] = trained_model
                        test_maes[name] = mae
                else:
                    with st.spinner(f"Entraînement de {name}..."):
                        mae, r2, trained_model = eval_and_detect(globals()[f"{name.lower()}_cv"], X, y, name)
                        joblib.dump(trained_model, file_path)
                        results.append({
                            'Modèle': name,
                            'CV MAE': mae,
                            'Train R²': r2_score(scaler_y.inverse_transform(y.reshape(-1, 1)), scaler_y.inverse_transform(trained_model.predict(X).reshape(-1, 1)))
                        })
                        models[name] = trained_model
                        test_maes[name] = mae
        else:
            for name, file_path in model_files.items():
                with st.spinner(f"Entraînement de {name}..."):
                    mae, r2, trained_model = eval_and_detect(globals()[f"{name.lower()}_cv"], X, y, name)
                    joblib.dump(trained_model, file_path)
                    results.append({
                        'Modèle': name,
                        'CV MAE': mae,
                        'Train R²': r2_score(scaler_y.inverse_transform(y.reshape(-1, 1)), scaler_y.inverse_transform(trained_model.predict(X).reshape(-1, 1)))
                    })
                    models[name] = trained_model
                    test_maes[name] = mae
    else:
        for name, file_path in model_files.items():
            with st.spinner(f"Entraînement de {name}..."):
                mae, r2, trained_model = eval_and_detect(globals()[f"{name.lower()}_cv"], X, y, name)
                joblib.dump(trained_model, file_path)
                results.append({
                    'Modèle': name,
                    'CV MAE': mae,
                    'Train R²': r2_score(scaler_y.inverse_transform(y.reshape(-1, 1)), scaler_y.inverse_transform(trained_model.predict(X).reshape(-1, 1)))
                })
                models[name] = trained_model
                test_maes[name] = mae

    return results, models, test_maes

# Fonction d'évaluation et d'interprétation
def interpret_results(model_name, train_mae, test_mae, train_r2, test_r2):
    rel_error = test_mae / np.mean(scaler_y.inverse_transform(y.reshape(-1, 1)))
    st.markdown("#### 💡 Interprétation")
    st.write(f"**R² sur test :** {test_r2:.4f} — indique la qualité de généralisation.")
    st.write(f"**MAE absolue :** {test_mae:.0f} — pour un PIB moyen ~{np.mean(scaler_y.inverse_transform(y.reshape(-1, 1))):,.0f}, soit une erreur relative d’environ **{rel_error*100:.1f}%**.")
    diff_r2 = train_r2 - test_r2
    if diff_r2 > 0.15:
        st.error("⚠️ Écart important entre R² d'entraînement et de test → possible surapprentissage.")
    else:
        st.success("✅ Pas de signe évident de surapprentissage.")

    st.markdown("#### ✅ Conclusion")
    if test_r2 >= 0.96 and rel_error < 0.03:
        st.write(f"✔️ **{model_name} donne d’excellents résultats.**")
        st.write("- Peut être utilisé comme benchmark.")
        st.write("- Très fiable pour un usage en prévision du PIB.")
    elif test_r2 >= 0.90:
        st.write(f"✔️ **{model_name} est un bon modèle,** mais peut être amélioré.")
    else:
        st.write(f"❌ **{model_name} montre des limites.** Envisagez une autre méthode ou un réglage plus poussé.")

def eval_and_detect(model_cv, X, y, model_name):
    model_cv.fit(X, y)
    train_pred = model_cv.predict(X)
    train_pred_unscaled = scaler_y.inverse_transform(train_pred.reshape(-1, 1)).flatten()
    y_unscaled = scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()
    train_mae = mean_absolute_error(y_unscaled, train_pred_unscaled)
    train_r2 = r2_score(y_unscaled, train_pred_unscaled)

    preds_test = []
    for tr, te in tscv.split(X):
        best_model = model_cv.best_estimator_
        best_model.fit(X[tr], y[tr])
        preds_test.extend(best_model.predict(X[te]))

    test_pred_unscaled = scaler_y.inverse_transform(np.array(preds_test).reshape(-1, 1)).flatten()
    test_mae = mean_absolute_error(y_unscaled[-len(preds_test):], test_pred_unscaled)
    test_r2 = r2_score(y_unscaled[-len(preds_test):], test_pred_unscaled)

    st.markdown(f"### 🔍 Résultats pour **{model_name}**")
    st.write(f"MAE d'entraînement : {train_mae:.2f}, MAE de test (TimeSeriesSplit) : {test_mae:.2f}")
    st.write(f"R² d'entraînement : {train_r2:.4f}, R² de test : {test_r2:.4f}")
    st.write(f"Meilleurs hyperparamètres : {model_cv.best_params_}")

    interpret_results(model_name, train_mae, test_mae, train_r2, test_r2)
    return test_mae, test_r2, model_cv

# Exécution des modèles
st.header("📊 Diagnostic et interprétation des modèles")
results, models, test_maes = load_or_train_models(X, y, cache_key)

if not test_maes:
    st.error("Erreur : Aucun modèle n'a été entraîné ou chargé. Veuillez vérifier les données d'entrée.")
    st.stop()

# Sélection du meilleur modèle
best_model_name = min(test_maes, key=test_maes.get)
best_model = models[best_model_name].best_estimator_
st.markdown(f"### 🏆 Modèle sélectionné : **{best_model_name}**")
st.write(f"Le modèle **{best_model_name}** a été choisi car il a le MAE le plus bas : {test_maes[best_model_name]:.2f}")

# Vérification du modèle sélectionné
st.header("🔎 Vérification du modèle sélectionné")
st.markdown("#### 1. Vérification de l'intégrité des données")
if X_df.isna().any().any():
    error_log.append("Valeurs manquantes détectées dans X_df.")
    st.error("Erreur : Valeurs manquantes dans les données d'entrée. Remplacement par 0.")
    X_df = X_df.fillna(0)
if y_df.isna().any().any():
    error_log.append("Valeurs manquantes détectées dans y_df.")
    st.warning("Attention : Valeurs manquantes dans les données cibles. Remplacement par la moyenne.")
    y_df = y_df.fillna(y_df.mean())
if y_df.empty or y_df.shape[0] == 0:
    error_log.append("y_df est vide ou n'a aucune ligne.")
    st.error("Erreur : Les données cibles (y_df) sont vides. Arrêt du programme.")
    st.stop()
st.success("Succès : Aucune valeur manquante dans les données après prétraitement. Forme de y_df : {}".format(y_df.shape))

st.markdown("#### 2. Vérification sur un ensemble de test")
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
best_model.fit(X_train, y_train)
y_pred_test = best_model.predict(X_test)
y_pred_test_unscaled = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()
y_test_unscaled = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
test_mae = mean_absolute_error(y_test_unscaled, y_pred_test_unscaled)
test_r2 = r2_score(y_test_unscaled, y_pred_test_unscaled)
st.write(f"MAE sur l'ensemble de test : {test_mae:.2f}")
st.write(f"R² sur l'ensemble de test : {test_r2:.4f}")
if test_mae > 1.5 * test_maes[best_model_name]:
    error_log.append(f"MAE sur l'ensemble de test ({test_mae:.2f}) significativement plus élevé que le MAE CV ({test_maes[best_model_name]:.2f}).")
    st.warning("Attention : Performance sur l'ensemble de test moins bonne que prévue.")

st.markdown("#### 3. Analyse des résidus")
residuals = y_test_unscaled - y_pred_test_unscaled
fig_residuals = px.scatter(
    x=range(len(residuals)),
    y=residuals,
    title="Résidus sur l'ensemble de test",
    labels={'x': 'Index', 'y': 'Résidus (million TND)'},
    color_discrete_sequence=['#FF6B6B'],
    render_mode='webgl'
)
fig_residuals.add_hline(y=0, line_dash="dash", line_color="black")
st.plotly_chart(fig_residuals, use_container_width=True)
if np.abs(residuals).mean() > test_maes[best_model_name]:
    error_log.append(f"Les résidus moyens ({np.abs(residuals).mean():.2f}) sont élevés par rapport au MAE CV ({test_maes[best_model_name]:.2f}).")
    st.warning("Attention : Les résidus montrent une erreur moyenne élevée, indiquant une possible sous-performance.")

st.markdown("#### 4. Intervalles de prédiction")
n_bootstraps = 50
bootstrap_preds = []
for _ in range(n_bootstraps):
    indices = np.random.choice(len(X_train), len(X_train), replace=True)
    best_model.fit(X_train[indices], y_train[indices])
    pred = best_model.predict(X_test)
    bootstrap_preds.append(scaler_y.inverse_transform(pred.reshape(-1, 1)).flatten())
bootstrap_preds = np.array(bootstrap_preds)
lower_bound = np.percentile(bootstrap_preds, 2.5, axis=0)
upper_bound = np.percentile(bootstrap_preds, 97.5, axis=0)
st.write("Intervalles de prédiction à 95% pour l'ensemble de test :")
for i, (lower, upper, actual) in enumerate(zip(lower_bound, upper_bound, y_test_unscaled)):
    st.write(f"Trimestre {i+1}: Prédit = {y_pred_test_unscaled[i]:,.0f}, Intervalle = [{lower:,.0f}, {upper:,.0f}], Réel = {actual:,.0f}")

# Prédiction pour 2024
if st.button("🔮 Prédire le PIB pour 2024"):
    with st.spinner("Entraînement et prédiction..."):
        start_time = time.time()
        quarters_to_predict = ["Q1", "Q2", "Q3", "Q4"]
        base_quarter = max(X_df.index)
        historical_df = pd.DataFrame({'Trimestre': [str(q) for q in quarters], 'PIB': scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()})
        pred_df = pd.DataFrame({'Trimestre': [f"{q} 2024" for q in quarters_to_predict], 'PIB': [0.0] * 4})
        combined_df = pd.concat([historical_df, pred_df], ignore_index=True)

        quarterly_predictions = []
        feature_vectors = []
        current_base_quarter = base_quarter
        current_base_data = X_df.loc[current_base_quarter][expected_features].copy()

        recent_data = X_df[expected_features].tail(4)
        growth_rates = {}
        for col in sectors + macro_rates:
            if col in recent_data.columns:
                quarter_growth = recent_data[col].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
                growth_rates[col] = quarter_growth.mean() * 100 if not quarter_growth.empty else 0.0
            else:
                growth_rates[col] = 0.0
                error_log.append(f"Taux de croissance pour '{col}' non calculé (colonne absente). Utilisation de 0.")
        for event in events:
            if event in recent_data.columns:
                growth_rates[event] = recent_data[event].iloc[-1] if not recent_data[event].empty else 0
            else:
                growth_rates[event] = 0
                error_log.append(f"Valeur pour '{event}' non trouvée. Utilisation de 0.")

        for i, q in enumerate(quarters_to_predict):
            feature_vector = pd.DataFrame(0.0, index=[0], columns=expected_features)
            logger.info(f"Création du vecteur de features pour {q} avec {len(feature_vector.columns)} colonnes : {list(feature_vector.columns)}")

            for sector in sectors:
                feature_vector[sector] = (
                    current_base_data[sector] * (1 + growth_rates.get(sector, 0.0) / 100)
                    if sector in X_df.columns else 0.0
                )
                if sector not in X_df.columns:
                    error_log.append(f"Secteur '{sector}' non trouvé pour {q}. Utilisation de 0.")

            for rate in macro_rates:
                feature_vector[rate] = (
                    current_base_data[rate] * (1 + growth_rates.get(rate, 0.0) / 100)
                    if rate in X_df.columns else 0.0
                )
                if rate not in X_df.columns:
                    error_log.append(f"Rate '{rate}' non trouvé pour {q}. Utilisation de 0.")

            for event in events:
                feature_vector[event] = growth_rates.get(event, 0.0)
                if event not in X_df.columns:
                    error_log.append(f"Événement '{event}' non trouvé pour {q}. Utilisation de 0.")

            for col in expected_features:
                if col.endswith('_lag1'):
                    base_col = col.replace('_lag1', '')
                    feature_vector[col] = (
                        current_base_data.get(base_col, X_df[base_col].mean() if base_col in X_df.columns else 0.0)
                        if base_col in feature_vector.columns else
                        current_base_data.get(col, X_df[col].mean() if col in X_df.columns else 0.0)
                    )
                    if feature_vector[col].iloc[0] == 0.0:
                        error_log.append(f"Feature décalée '{col}' pour {q} définie à 0.")

            if list(feature_vector.columns) != expected_features:
                error_log.append(f"Non-concordance dans les colonnes de feature_vector pour {q}: Attendu {len(expected_features)} colonnes, Obtenu {len(feature_vector.columns)} colonnes: {list(feature_vector.columns)}")
                st.error(f"Erreur : Les colonnes de feature_vector ({len(feature_vector.columns)}) ne correspondent pas à expected_features ({len(expected_features)}).")
                st.stop()

            if feature_vector.isna().any().any():
                error_log.append(f"Valeurs NaN pour {q} : {feature_vector.columns[feature_vector.isna().any()].tolist()}. Remplacement par 0.")
                feature_vector = feature_vector.fillna(0.0)

            X_new = scaler_X.transform(feature_vector)
            feature_vectors.append(X_new)

            predicted_gdp = float(scaler_y.inverse_transform(best_model.predict(X_new).reshape(-1, 1))[0])
            quarterly_predictions.append(predicted_gdp)
            combined_df.loc[combined_df['Trimestre'] == f"{q} 2024", 'PIB'] = predicted_gdp

            current_base_data = feature_vector.iloc[0].copy()

        yearly_gdp = sum(quarterly_predictions)
        st.markdown("### 📈 Résultat de la prédiction")
        st.write(f"**Modèle utilisé :** {best_model_name}")
        st.write("**Prédictions trimestrielles :**")
        for i, q in enumerate(quarters_to_predict):
            st.write(f"- **{q} 2024** : {quarterly_predictions[i]:,.0f} million TND")
        st.write(f"**PIB annuel estimé pour 2024** : {yearly_gdp:,.0f} million TND")

        y_pred_historical = best_model.predict(X)
        y_pred_historical_unscaled = scaler_y.inverse_transform(y_pred_historical.reshape(-1, 1)).flatten()
        y_historical_unscaled = scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()
        historical_df = pd.DataFrame({
            'Trimestre': [str(q) for q in quarters],
            'PIB Réel': y_historical_unscaled,
            'PIB Prédit': y_pred_historical_unscaled
        })
        pred_df = pd.DataFrame({
            'Trimestre': [f"{q} 2024" for q in quarters_to_predict],
            'PIB Réel': [np.nan] * 4,
            'PIB Prédit': quarterly_predictions
        })
        combined_df = pd.concat([historical_df, pred_df], ignore_index=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=combined_df['Trimestre'],
            y=combined_df['PIB Réel'],
            mode='lines+markers',
            name='PIB Réel',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=combined_df['Trimestre'],
            y=combined_df['PIB Prédit'],
            mode='lines+markers',
            name='PIB Prédit',
            line=dict(color='red', dash='dash')
        ))
        fig.update_layout(
            title='PIB Historique vs Prédictions (incl. 2024)',
            xaxis_title='Trimestre',
            yaxis_title='PIB (million TND)',
            xaxis_tickangle=45,
            height=500,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🧠 Explication des prédictions avec SHAP")
        st.write("Les graphiques suivants expliquent comment chaque feature contribue aux prédictions du PIB pour 2024.")
        best_model.fit(X, y)
        feature_vectors_for_shap = np.vstack(feature_vectors)
        error_log.append(f"Forme de feature_vectors_for_shap : {feature_vectors_for_shap.shape}")
        background_data = scaler_X.transform(X_df[expected_features].iloc[:20])
        error_log.append(f"Forme de background_data : {background_data.shape}")

        try:
            if best_model_name in ["Ridge", "ElasticNet"]:
                explainer = shap.LinearExplainer(
                    best_model,
                    background_data,
                    feature_names=expected_features
                )
            else:
                explainer = shap.KernelExplainer(
                    best_model.predict,
                    background_data,
                    feature_names=expected_features,
                    nsamples=100
                )

            shap_values = explainer.shap_values(feature_vectors_for_shap)
            error_log.append(f"Forme de shap_values : {np.array(shap_values).shape}")

            st.markdown("#### 📊 Importance globale des features (Summary Plot)")
            plt.figure(figsize=(10, 6), dpi=80)
            shap.summary_plot(shap_values, feature_vectors_for_shap, feature_names=expected_features, show=False)
            st.pyplot(plt)
            plt.close()

            st.markdown("#### 📊 Importance des features par trimestre")
            for i, q in enumerate(quarters_to_predict):
                st.write(f"**{q} 2024**")
                plt.figure(figsize=(10, 6), dpi=80)
                shap.bar_plot(shap_values[i], feature_names=expected_features, max_display=10, show=False)
                st.pyplot(plt)
                plt.close()

        except Exception as e:
            error_log.append(f"Erreur lors du calcul SHAP : {str(e)}")
            st.error(f"Erreur : Impossible de générer les explications SHAP : {str(e)}. Veuillez vérifier les données.")

        st.info(f"🧪 Prédiction basée sur le modèle {best_model_name} avec le MAE le plus bas, utilisant les tendances historiques extrapolées à partir des 4 derniers trimestres.")
        logger.info(f"Prédiction terminée en {time.time() - start_time:.2f} secondes")

        show_errors = st.checkbox("Afficher le journal", value=True)
        if show_errors and error_log:
            st.markdown("### Journal informatif")
            for error in error_log:
                st.write(error)
