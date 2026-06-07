"""
Pydantic Schemas for Churn Prediction API
Author: Yuvraaj M N
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
from enum import Enum


class GenderEnum(str, Enum):
    male = "Male"
    female = "Female"


class ContractEnum(str, Enum):
    month_to_month = "Month-to-month"
    one_year = "One year"
    two_year = "Two year"


class InternetServiceEnum(str, Enum):
    dsl = "DSL"
    fiber_optic = "Fiber optic"
    no = "No"


class PaymentMethodEnum(str, Enum):
    electronic_check = "Electronic check"
    mailed_check = "Mailed check"
    bank_transfer = "Bank transfer (automatic)"
    credit_card = "Credit card (automatic)"


class CustomerFeatures(BaseModel):
    """Input features for churn prediction"""

    # Demographics
    gender: GenderEnum = Field(..., example="Male")
    SeniorCitizen: int = Field(..., ge=0, le=1, example=0)
    Partner: Literal["Yes", "No"] = Field(..., example="Yes")
    Dependents: Literal["Yes", "No"] = Field(..., example="No")

    # Service info
    tenure: int = Field(..., ge=0, le=72, example=12)
    PhoneService: Literal["Yes", "No"] = Field(..., example="Yes")
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(..., example="No")
    InternetService: InternetServiceEnum = Field(..., example="Fiber optic")
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(..., example="No")
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(..., example="Yes")
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(..., example="No")
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(..., example="No")
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(..., example="Yes")
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(..., example="No")

    # Contract & billing
    Contract: ContractEnum = Field(..., example="Month-to-month")
    PaperlessBilling: Literal["Yes", "No"] = Field(..., example="Yes")
    PaymentMethod: PaymentMethodEnum = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., ge=0, le=200, example=70.35)
    TotalCharges: float = Field(..., ge=0, example=150.30)

    class Config:
        use_enum_values = True


class PredictionResponse(BaseModel):
    """Response schema for prediction"""
    model_config = {"protected_namespaces": ()}
    customer_id: Optional[str] = None
    churn_prediction: int
    churn_probability: float
    churn_label: str
    risk_level: str
    model_used: str
    model_version: str
    cached: bool
    confidence: str


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    customers: list[CustomerFeatures]
    customer_ids: Optional[list[str]] = None


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: list[PredictionResponse]
    total: int
    churn_count: int
    churn_rate: float


class HealthResponse(BaseModel):
    """Health check response"""
    model_config = {"protected_namespaces": ()}
    status: str
    model_loaded: bool
    redis_connected: bool
    champion_model: str
    challenger_model: str
    version: str


class ModelMetrics(BaseModel):
    """Model performance metrics"""
    model_config = {"protected_namespaces": ()}
    model_name: str
    accuracy: float
    roc_auc: float
    f1: float
    precision: float
    recall: float