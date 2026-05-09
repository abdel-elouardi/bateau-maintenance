from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib

# ← CHANGER SI BESOIN
MODEL_PATH  = 'models/model.pkl'
SCALER_PATH = 'models/scaler.pkl'

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

app = FastAPI(title="ML Template API")

class Donnees(BaseModel):
    features: List[float]

@app.get("/")
def accueil():
    return {"status": "ok"}

@app.post("/predict")
def predire(data: Donnees):
    X    = np.array(data.features).reshape(1, -1)
    X    = scaler.transform(X)
    pred = model.predict(X)[0]
    return {"prediction": float(pred)}