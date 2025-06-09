from fastapi import FastAPI, HTTPException
from model_loader import load_latest_model
import pandas as pd
import logging

app = FastAPI()
logger = logging.getLogger(__name__)

model = None
current_run_id = None

@app.on_event("startup")
async def startup_event():
    global model, current_run_id
    try:
        # ici on a bien MLFLOW_TRACKING_URI dans l'env
        model, current_run_id = load_latest_model()
        logger.info(f"Loaded MLflow model from run {current_run_id!r}")
    except Exception as e:
        # on affiche le problème mais on ne crash pas tout
        logger.warning(f"Could not load MLflow model at startup: {e!r}")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: dict):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    df = pd.DataFrame([payload])
    pred = model.predict(df)[0]
    return {"run_id": current_run_id, "prediction": int(pred)}
