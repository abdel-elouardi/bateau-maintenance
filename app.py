import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ← CHANGER CES LIGNES
TITRE = "ML Dashboard"
CIBLE = 'target'

model  = joblib.load('models/model.pkl')
scaler = joblib.load('models/scaler.pkl')

st.set_page_config(page_title=TITRE, page_icon="🤖", layout="wide")
st.title(f"🤖 {TITRE}")
st.markdown("---")

page = st.sidebar.radio("Navigation", ["📊 Données", "🔮 Prédiction"])

if page == "📊 Données":
    st.header("📊 Exploration des données")
    fichier = st.file_uploader("Charger un CSV", type=["csv"])
    if fichier:
        df = pd.read_csv(fichier)
        st.success(f"✅ {df.shape[0]} lignes, {df.shape[1]} colonnes")
        st.dataframe(df.head())
        st.dataframe(df.describe())

elif page == "🔮 Prédiction":
    st.header("🔮 Faire une prédiction")
    features_input = st.text_input("Features (séparées par des virgules)")
    if st.button("🔮 Prédire", type="primary"):
        try:
            features = [float(x.strip()) for x in features_input.split(",")]
            X    = np.array(features).reshape(1, -1)
            X    = scaler.transform(X)
            pred = model.predict(X)[0]
            st.success(f"✅ Prédiction : {pred}")
        except Exception as e:
            st.error(f"❌ Erreur : {e}")