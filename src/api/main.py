"""
FastAPI Churn Prediction API
Endpoints: /predict, /predict/batch, /health, /ab-test/report, /cache/stats
Author: Yuvraaj M N
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.schemas import (
    CustomerFeatures, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    HealthResponse
)
from src.api.cache import cache
from src.api.ab_test import ab_manager

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# PATHS
# ============================================
PROCESSED_PATH = Path("data/processed")
MODELS_PATH = Path("models")

# ============================================
# LOAD MODELS + PREPROCESSOR
# ============================================
print("\n🔧 Loading models...")

preprocessor = joblib.load(PROCESSED_PATH / "preprocessor.pkl")
print("   ✅ Preprocessor loaded")

models = {}
for model_name in ["logistic_regression", "xgboost", "lightgbm"]:
    model_path = MODELS_PATH / f"{model_name}.pkl"
    if model_path.exists():
        models[model_name] = joblib.load(model_path)
        print(f"   ✅ {model_name} loaded")

# Load model info
with open(MODELS_PATH / "best_model_info.json") as f:
    model_info = json.load(f)

print(f"   🏆 Champion: {model_info['champion']}")
print(f"   🥊 Challenger: {model_info['challenger']}\n")

# ============================================
# FASTAPI APP
# ============================================
app = FastAPI(
    title="🤖 Churn Predictor API",
    description="""
## Telco Customer Churn Prediction API

### Models
- **Baseline**: Logistic Regression (AUC: 0.845) — reference model
- **Champion**: XGBoost (AUC: 0.842, Recall: **82.1%**) — production deployed
- **Challenger**: LightGBM (AUC: 0.843) — A/B test (20% traffic)

### Why XGBoost as Champion?
In churn prediction, **recall matters most**.
Missing a churner = losing a customer forever.
XGBoost catches 82.1% of churners vs LightGBM's 60.7%.

### Features
- **A/B Testing**: XGBoost (80%) vs LightGBM (20%)
- **Redis Caching**: TTL=1 hour, invalidated on model update
- **Baseline Comparison**: All predictions benchmarked vs LR
    """,
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# HELPER FUNCTIONS
# ============================================
def features_to_dataframe(customer: CustomerFeatures) -> pd.DataFrame:
    """Convert Pydantic model to DataFrame"""
    data = customer.dict()
    df = pd.DataFrame([data])
    return df


def engineer_features_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Apply same feature engineering as training"""

    # Tenure group
    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 60, 72],
        labels=['0-1yr', '1-2yr', '2-4yr', '4-5yr', '5-6yr']
    )

    # Avg monthly vs total
    df['avg_monthly_vs_total'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)

    # Number of services
    service_cols = [
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    df['num_services'] = df[service_cols].apply(
        lambda row: sum(1 for v in row
                        if v not in ['No', 'No internet service', 'No phone service']),
        axis=1
    )

    # Binary features
    df['is_high_value'] = (df['MonthlyCharges'] > 64.76).astype(int)
    df['is_long_term'] = (df['tenure'] > 24).astype(int)

    # Contract risk
    contract_risk = {'Month-to-month': 2, 'One year': 1, 'Two year': 0}
    df['contract_risk'] = df['Contract'].map(contract_risk)

    return df


def get_risk_level(probability: float) -> str:
    """Convert probability to risk level"""
    if probability >= 0.7:
        return "HIGH"
    elif probability >= 0.4:
        return "MEDIUM"
    else:
        return "LOW"


def make_prediction(
    customer: CustomerFeatures,
    model_name: str,
    customer_id: Optional[str] = None
) -> dict:
    """Core prediction function"""

    features_dict = customer.dict()

    # Check cache first
    cached_result = cache.get(features_dict, model_name)
    if cached_result:
        cached_result["cached"] = True
        return cached_result

    # Convert to DataFrame
    df = features_to_dataframe(customer)

    # Feature engineering
    df = engineer_features_for_inference(df)

    # Preprocess
    X = preprocessor.transform(df)

    # Get model
    model_key = model_name.lower().replace(" ", "_").replace("-", "_")
    if model_key not in models:
        model_key = "xgboost"  # fallback

    model = models[model_key]

    # Predict
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])

    result = {
        "customer_id": customer_id,
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4),
        "churn_label": "Will Churn" if prediction == 1 else "Will Stay",
        "risk_level": get_risk_level(probability),
        "model_used": model_name,
        "model_version": "2.0",
        "cached": False,
        "confidence": "HIGH" if probability > 0.8 or probability < 0.2 else "MEDIUM"
    }

    # Cache result
    cache.set(features_dict, model_name, result)

    return result


# ============================================
# ENDPOINTS
# ============================================

@app.get("/", include_in_schema=False)
def root():
    return {"message": "Churn Predictor API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=len(models) > 0,
        redis_connected=cache.connected,
        champion_model=model_info["champion"],
        challenger_model=model_info["challenger"],
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(
    customer: CustomerFeatures,
    customer_id: Optional[str] = Query(None, description="Customer ID for consistent A/B assignment"),
    use_champion: bool = Query(False, description="Force champion model")
):
    """
    Predict churn for a single customer.

    - **A/B Testing**: 80% champion, 20% challenger (unless force_champion=True)
    - **Caching**: Results cached for 1 hour (TTL configurable)
    - **Risk Levels**: LOW (<40%), MEDIUM (40-70%), HIGH (>70%)
    """
    try:
        # A/B test model selection
        if use_champion:
            model_name = model_info["champion"].lower().replace(" ", "_")
            variant = "champion"
        else:
            model_name, variant = ab_manager.select_model(customer_id)

        # Make prediction
        result = make_prediction(customer, model_name, customer_id)

        # Record A/B result
        ab_manager.record_result(variant, result["churn_probability"])

        logger.info(f"✅ Prediction: {result['churn_label']} "
                   f"(prob={result['churn_probability']:.3f}, "
                   f"model={model_name}, cached={result['cached']})")

        return PredictionResponse(**result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(request: BatchPredictionRequest):
    """
    Predict churn for multiple customers at once.
    Maximum 100 customers per request.
    """
    if len(request.customers) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 customers per batch request"
        )

    try:
        predictions = []
        model_name = model_info["champion"].lower().replace(" ", "_")

        for i, customer in enumerate(request.customers):
            customer_id = None
            if request.customer_ids and i < len(request.customer_ids):
                customer_id = request.customer_ids[i]

            result = make_prediction(customer, model_name, customer_id)
            predictions.append(PredictionResponse(**result))

        churn_count = sum(1 for p in predictions if p.churn_prediction == 1)

        return BatchPredictionResponse(
            predictions=predictions,
            total=len(predictions),
            churn_count=churn_count,
            churn_rate=round(churn_count / len(predictions), 4)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ab-test/report", tags=["A/B Testing"])
def get_ab_report():
    """
    Get A/B test report comparing champion vs challenger.
    Auto rollback triggered if challenger diverges >5%.
    """
    return ab_manager.get_report()


@app.get("/cache/stats", tags=["Cache"])
def get_cache_stats():
    """Get Redis cache statistics"""
    return cache.get_stats()


@app.delete("/cache/invalidate", tags=["Cache"])
def invalidate_cache(model_name: str = Query(...)):
    """
    Invalidate cached predictions for a model.
    Called automatically on model version bump via MLflow webhook.
    """
    count = cache.invalidate_model(model_name)
    return {"invalidated_keys": count, "model": model_name}


@app.get("/models", tags=["Models"])
def list_models():
    """List all models with their roles and metrics"""
    return {
        "baseline": {
            "name": "Logistic Regression",
            "role": "baseline",
            "metrics": model_info["baseline_metrics"],
            "note": "Simple interpretable reference model"
        },
        "champion": {
            "name": "XGBoost",
            "role": "champion — production deployed",
            "metrics": model_info["champion_metrics"],
            "note": "Best recall (82.1%) — catches most churners"
        },
        "challenger": {
            "name": "LightGBM",
            "role": "challenger — 20% A/B traffic",
            "metrics": model_info["challenger_metrics"],
            "note": "Best accuracy (78.6%) — being tested"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )