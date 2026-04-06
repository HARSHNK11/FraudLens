"""
FraudLens FastAPI Backend
Endpoints:
  GET  /                  Health check
  GET  /model/info        Model name, metrics, threshold
  POST /predict           Single transaction fraud prediction + SHAP
  POST /predict/batch     Batch prediction
  GET  /plots/{filename}  Serve training visualization PNGs
"""

import os
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(ROOT, "artifacts", "plots")

from backend.schemas import (
    TransactionInput, PredictionResponse,
    ModelInfoResponse, BatchTransactionInput, BatchPredictionResponse,
)
from backend.model_loader import registry
from backend.shap_explainer import compute_shap_values


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML artifacts once at startup."""
    registry.load()
    yield


app = FastAPI(
    title="FraudLens API",
    description="Real-time credit card fraud detection with ML + SHAP explainability",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount plots directory for static image serving
os.makedirs(PLOTS_DIR, exist_ok=True)
app.mount("/plots", StaticFiles(directory=PLOTS_DIR), name="plots")


# ─── Helpers ───────────────────────────────────────────────────────────────────
def transaction_to_array(tx: TransactionInput) -> np.ndarray:
    """Convert Pydantic model to numpy array in correct feature order dynamically."""
    tx_dict = tx.model_dump()
    # Default to 0.0 for any missing features
    arr = [tx_dict.get(f) if tx_dict.get(f) is not None else 0.0 for f in registry.features]
    return np.array(arr, dtype=float)


def scale_row(raw: np.ndarray) -> np.ndarray:
    """Scale Amount and Time using the fitted scaler, locating their dynamic indices."""
    scaled = raw.copy()
    
    amount_idx = registry.features.index("Amount") if "Amount" in registry.features else -1
    time_idx = registry.features.index("Time") if "Time" in registry.features else -1
    
    amount_val = raw[amount_idx] if amount_idx != -1 else 0.0
    time_val = raw[time_idx] if time_idx != -1 else 0.0
    
    # scaler was fit on X[["Amount", "Time"]] in train.py
    scaled_at = registry.scaler.transform([[amount_val, time_val]])[0]
    
    if amount_idx != -1:
        scaled[amount_idx] = scaled_at[0]
    if time_idx != -1:
        scaled[time_idx] = scaled_at[1]
        
    return scaled


def get_risk_level(prob: float) -> str:
    if prob < 0.3:
        return "LOW"
    elif prob < 0.6:
        return "MEDIUM"
    elif prob < 0.85:
        return "HIGH"
    else:
        return "CRITICAL"


def make_prediction(tx: TransactionInput, include_shap: bool = True) -> PredictionResponse:
    raw = transaction_to_array(tx)
    scaled = scale_row(raw)

    prob = float(registry.model.predict_proba([scaled])[0, 1])
    is_fraud = prob >= registry.threshold
    risk = get_risk_level(prob)

    shap_vals = {}
    top_features = []
    if include_shap:
        shap_vals = compute_shap_values(registry.model, scaled, registry.features)
        top_features = [
            {"feature": k, "value": round(v, 6), "direction": "fraud" if v > 0 else "legit"}
            for k, v in sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)
        ]

    return PredictionResponse(
        is_fraud=is_fraud,
        fraud_probability=round(prob, 6),
        risk_level=risk,
        threshold_used=round(registry.threshold, 6),
        shap_values=shap_vals if shap_vals else None,
        top_features=top_features if top_features else None,
    )


# ─── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", summary="Health Check")
def root():
    return {
        "status": "ok",
        "service": "FraudLens API",
        "model": registry.model_name,
    }


@app.get("/model/info", response_model=ModelInfoResponse, summary="Model Info")
def model_info():
    return ModelInfoResponse(
        model_name=registry.model_name,
        threshold=round(registry.threshold, 6),
        metrics=registry.metrics,
        features=registry.features,
    )


@app.get("/model/log", summary="Get Model Training Log")
def get_model_log():
    log_path = os.path.join(ROOT, "artifacts", "training.log")
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            return {"log": f.read()}
    return {"log": "No training log found. Run ml/train.py first."}


@app.get("/models/all", summary="Get All Models Metrics")
def get_all_models():
    path = os.path.join(ROOT, "artifacts", "all_models_info.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@app.post("/predict", response_model=PredictionResponse, summary="Predict Fraud")
def predict(transaction: TransactionInput):
    if registry.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")
    return make_prediction(transaction, include_shap=True)


@app.post("/predict/batch", response_model=BatchPredictionResponse, summary="Batch Predict")
def predict_batch(body: BatchTransactionInput):
    if registry.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if len(body.transactions) > 1000:
        raise HTTPException(status_code=400, detail="Batch size limit is 1000.")

    predictions = [make_prediction(tx, include_shap=False) for tx in body.transactions]
    fraud_count = sum(1 for p in predictions if p.is_fraud)

    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        fraud_count=fraud_count,
    )
