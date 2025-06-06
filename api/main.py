from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from model_loader import load_latest_model

app = FastAPI(title="Fraud-Score API")

model, current_run_id = load_latest_model()

class Transaction(BaseModel):
    amount: float
    merchant_id: int
    country: str
    card_age_months: int
    # … ajoute les features que ton modèle attend

@app.get("/health")
def health():
    return {"status": "ok", "model_run": current_run_id}

@app.post("/predict")
def predict(tx: Transaction):
    df = pd.DataFrame([tx.dict()])
    proba = model.predict_proba(df)[:, 1][0]  # classe 1 = fraude
    return {
        "fraud_probability": round(float(proba), 4),
        "model_run": current_run_id
    }
