"""
FastAPI Endpoint Tests
Author: Yuvraaj M N
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

SAMPLE_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 150.30
}

LOW_RISK_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 60,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 45.50,
    "TotalCharges": 2700.0
}


def test_health_endpoint():
    """Health check should return healthy"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] == True


def test_predict_endpoint():
    """Prediction endpoint should return valid response"""
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200
    data = response.json()

    assert "churn_prediction" in data
    assert "churn_probability" in data
    assert "churn_label" in data
    assert "risk_level" in data
    assert "model_used" in data

    assert data["churn_prediction"] in [0, 1]
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]


def test_high_risk_prediction():
    """Month-to-month Fiber optic customer should predict churn"""
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    # Should predict churn
    assert data["churn_prediction"] == 1
    # Should be MEDIUM or HIGH risk (not LOW)
    assert data["risk_level"] in ["MEDIUM", "HIGH"]
    assert data["churn_probability"] > 0.4


def test_low_risk_prediction():
    """Long-tenure two-year contract customer should be LOW risk"""
    response = client.post(
        "/predict?use_champion=true",
        json=LOW_RISK_CUSTOMER
    )
    assert response.status_code == 200
    data = response.json()
    assert data["churn_prediction"] == 0
    assert data["churn_probability"] < 0.5


def test_batch_predict():
    """Batch prediction should work"""
    response = client.post(
        "/predict/batch",
        json={"customers": [SAMPLE_CUSTOMER, LOW_RISK_CUSTOMER]}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert "churn_rate" in data
    assert "predictions" in data


def test_models_endpoint():
    """Models endpoint should return all three models"""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "champion" in data
    assert "challenger" in data
    assert "baseline" in data


def test_ab_report():
    """A/B test report should return valid data"""
    response = client.get("/ab-test/report")
    assert response.status_code == 200
    data = response.json()
    assert "champion" in data
    assert "challenger" in data
    assert "results" in data


def test_invalid_input():
    """Invalid input should return 422"""
    response = client.post("/predict", json={"invalid": "data"})
    assert response.status_code == 422


def test_cache_stats():
    """Cache stats endpoint should work"""
    response = client.get("/cache/stats")
    assert response.status_code == 200