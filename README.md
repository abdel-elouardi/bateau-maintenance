# 🚢 Bateau — Détection de Pannes Moteurs Marins

Système ML de détection de pannes sur moteurs marins.
Applicable chez Bénéteau, Strato Compo, Naval Group.

## 📊 Dataset
- Source : Marine Engine Performance & Fault Diagnosis
- Taille : 10 000 mesures
- Capteurs : RPM, température huile, pression, vibrations, cylindres
- Classes : 8 types (Normal + 7 types de pannes)

## 🏆 Résultats
| Modèle | Accuracy |
|--------|---------|
| LightGBM | 88.6% 🏆 |
| Random Forest | 88.35% |
| SVM | 88.35% |
| XGBoost | 88.0% |
| Gradient Boosting | 87.85% |
| Logistic Regression | 85.25% |
| KNN | 85.0% |

## 📊 Détection par type de panne
| Panne | F1-Score |
|-------|---------|
| Normal | 92% |
| Panne type 4 | 100% 🏆 |
| Panne type 7 | 100% 🏆 |
| Panne type 5 | 95% |
| Panne type 6 | 97% |

## 🚀 Lancement
```bash
pip install -r requirements.txt
python3 main.py
uvicorn api:app --port 8000
streamlit run app.py
```

## 🛠️ Technologies
Python 3.11 | Scikit-learn | LightGBM | XGBoost | FastAPI | Streamlit