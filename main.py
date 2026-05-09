import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from pathlib import Path

# ════════════════════════════════════
# ⚙️ CHANGER CES LIGNES SELON TON PROJET
# ════════════════════════════════════
FICHIER = 'data/raw/marine_engine_fault_dataset.csv'  # ← ton fichier CSV
CIBLE   = 'Fault_Label'                    # ← colonne à prédire
TACHE   = 'classification'        # ← 'classification' ou 'regression'

# ════════════════════════════════════
# 1. CHARGER
# ════════════════════════════════════
df = pd.read_csv(FICHIER)
print(f"✅ {df.shape[0]} lignes, {df.shape[1]} colonnes")
# Supprimer les colonnes dates
df = df.drop(columns=['Timestamp'])
# ════════════════════════════════════
# 2. NETTOYER
# ════════════════════════════════════
# Espaces
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

# Types
for col in df.columns:
    if df[col].dtype == object:
        try: df[col] = pd.to_numeric(df[col])
        except: pass

# Doublons
df = df.drop_duplicates()

# Colonnes > 75% vides
df = df.drop(columns=df.columns[df.isnull().mean() > 0.75])

# Valeurs manquantes
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].median() if abs(df[col].skew()) > 1 else df[col].mean())

# Colonnes constantes
cols_constantes = [col for col in df.columns if df[col].nunique() == 1]
df = df.drop(columns=cols_constantes)

# Skewness + Outliers
cols_num = [col for col in df.columns if col != CIBLE and df[col].dtype != object]
for col in cols_num:
    if df[col].skew() > 1:
        df[col] = np.log1p(df[col] - df[col].min())

# Corrélation > 0.9
cols_a_supprimer = []
corr = df[cols_num].corr().abs()
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        if corr.loc[corr.columns[i], corr.columns[j]] > 0.9:
            cols_a_supprimer.append(corr.columns[j])
df = df.drop(columns=list(set(cols_a_supprimer)))

# Encoding texte → chiffres
for col in df.select_dtypes(include="object").columns:
    if col != CIBLE:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

print(f"✅ Nettoyage : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# ════════════════════════════════════
# 2.5 FEATURE ENGINEERING (optionnel)
# Ajouter tes colonnes calculées ici
# ════════════════════════════════════
# Exemples :
# df['ratio']      = df['col1'] / df['col2']
# df['somme']      = df['col1'] + df['col2']
# df['difference'] = df['col1'] - df['col2']

# ════════════════════════════════════
# 3. SPLIT
# ════════════════════════════════════
X = df.drop(columns=[CIBLE])
y = df[CIBLE]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✅ Train:{len(X_train)} Test:{len(X_test)}")

# ════════════════════════════════════
# 4. NORMALISER
# ════════════════════════════════════
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
print("✅ Normalisation terminée")

# ════════════════════════════════════
# 5. MODÈLES
# ════════════════════════════════════
if TACHE == 'classification':
    models = {
        "Random Forest":       RandomForestClassifier(random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM":                 SVC(random_state=42),
        "KNN":                 KNeighborsClassifier(),
        "XGBoost":             XGBClassifier(random_state=42, eval_metric='mlogloss'),
        "LightGBM":            LGBMClassifier(random_state=42, verbose=-1),
    }
else:
    models = {
        "Random Forest":     RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Linear Regression": LinearRegression(),
        "KNN":               KNeighborsRegressor(),
        "XGBoost":           XGBRegressor(random_state=42),
        "LightGBM":          LGBMRegressor(random_state=42, verbose=-1),
    }

resultats = {}
for nom, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if TACHE == 'classification':
        score = accuracy_score(y_test, y_pred)
        resultats[nom] = score
        print(f"  {nom:25s} : {score:.4f}")
    else:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        resultats[nom] = rmse
        print(f"  {nom:25s} : RMSE {rmse:.2f}")

# ════════════════════════════════════
# 6. MEILLEUR MODÈLE
# ════════════════════════════════════
if TACHE == 'classification':
    meilleur = max(resultats, key=resultats.get)
    print(f"\n🏆 Meilleur : {meilleur} ({resultats[meilleur]:.4f})")
    print(classification_report(y_test, models[meilleur].predict(X_test)))
else:
    meilleur = min(resultats, key=resultats.get)
    print(f"\n🏆 Meilleur : {meilleur} (RMSE:{resultats[meilleur]:.2f})")

# ════════════════════════════════════
# 7. GRAPHIQUES
# ════════════════════════════════════
plt.style.use('dark_background')

# Comparaison des modèles
plt.figure(figsize=(10, 5))
sns.barplot(x=list(resultats.values()), y=list(resultats.keys()),
            hue=list(resultats.keys()), palette="Blues_r", legend=False)
plt.title("Comparaison des modeles")
plt.tight_layout()
Path("models").mkdir(exist_ok=True)
plt.savefig("models/comparaison.png")
print("✅ Graphique sauvegardé")

# ════════════════════════════════════
# 8. SAUVEGARDER
# ════════════════════════════════════
joblib.dump(models[meilleur], 'models/model.pkl')
joblib.dump(scaler,           'models/scaler.pkl')
print("✅ Modèle sauvegardé")