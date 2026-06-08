"""
Model Performance Tests
Author: Yuvraaj M N
"""

import pytest
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score

PROCESSED_PATH = Path("data/processed")
MODELS_PATH = Path("models")
REPORTS_PATH = Path("reports")

# Thresholds
MIN_AUC = 0.80
MIN_RECALL = 0.75
MIN_ACCURACY = 0.70


@pytest.fixture
def test_data():
    X_test = np.load(PROCESSED_PATH / "X_test.npy")
    y_test = np.load(PROCESSED_PATH / "y_test.npy")
    return X_test, y_test


@pytest.fixture
def xgboost_model():
    return joblib.load(MODELS_PATH / "xgboost.pkl")


@pytest.fixture
def lightgbm_model():
    return joblib.load(MODELS_PATH / "lightgbm.pkl")


def test_model_files_exist():
    """All model files should exist"""
    models = ["xgboost.pkl", "lightgbm.pkl", "logistic_regression.pkl"]
    for m in models:
        assert (MODELS_PATH / m).exists(), f"Missing model: {m}"


def test_xgboost_auc(xgboost_model, test_data):
    """XGBoost AUC should exceed threshold"""
    X_test, y_test = test_data
    y_prob = xgboost_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    assert auc >= MIN_AUC, f"XGBoost AUC {auc:.4f} below threshold {MIN_AUC}"
    print(f"XGBoost AUC: {auc:.4f} ✅")


def test_xgboost_recall(xgboost_model, test_data):
    """XGBoost recall should exceed threshold (catching churners)"""
    X_test, y_test = test_data
    y_pred = xgboost_model.predict(X_test)
    recall = recall_score(y_test, y_pred)
    assert recall >= MIN_RECALL, f"XGBoost Recall {recall:.4f} below threshold {MIN_RECALL}"
    print(f"XGBoost Recall: {recall:.4f} ✅")


def test_xgboost_accuracy(xgboost_model, test_data):
    """XGBoost accuracy should exceed threshold"""
    X_test, y_test = test_data
    y_pred = xgboost_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    assert acc >= MIN_ACCURACY, f"Accuracy {acc:.4f} below threshold {MIN_ACCURACY}"


def test_lightgbm_auc(lightgbm_model, test_data):
    """LightGBM AUC should exceed threshold"""
    X_test, y_test = test_data
    y_prob = lightgbm_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    assert auc >= MIN_AUC, f"LightGBM AUC {auc:.4f} below threshold {MIN_AUC}"


def test_champion_beats_baseline(test_data):
    """XGBoost recall should beat Logistic Regression recall"""
    X_test, y_test = test_data
    xgb = joblib.load(MODELS_PATH / "xgboost.pkl")
    lr = joblib.load(MODELS_PATH / "logistic_regression.pkl")

    xgb_recall = recall_score(y_test, xgb.predict(X_test))
    lr_recall = recall_score(y_test, lr.predict(X_test))

    assert xgb_recall >= lr_recall * 0.95, \
        f"Champion recall {xgb_recall:.4f} significantly below baseline {lr_recall:.4f}"
    print(f"XGBoost recall: {xgb_recall:.4f} vs LR recall: {lr_recall:.4f} ✅")


def test_model_info_exists():
    """Best model info JSON should exist"""
    info_path = MODELS_PATH / "best_model_info.json"
    assert info_path.exists()
    with open(info_path) as f:
        info = json.load(f)
    assert "champion" in info
    assert "challenger" in info
    assert info["champion"] == "XGBoost"


def test_shap_reports_exist():
    """SHAP reports should be generated"""
    assert (REPORTS_PATH / "shap_bar_xgboost.png").exists()
    assert (REPORTS_PATH / "top_features_xgboost.csv").exists()