"""
Data Validation Tests
Author: Yuvraaj M N
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_PATH = Path("data/processed")
RAW_PATH = Path("data/raw")


def test_raw_data_exists():
    """Raw data file should exist"""
    assert (RAW_PATH / "telco_churn.csv").exists()


def test_raw_data_shape():
    """Dataset should have correct shape"""
    df = pd.read_csv(RAW_PATH / "telco_churn.csv")
    assert df.shape[0] == 7043, f"Expected 7043 rows, got {df.shape[0]}"
    assert df.shape[1] == 21, f"Expected 21 columns, got {df.shape[1]}"


def test_raw_data_columns():
    """All required columns should be present"""
    df = pd.read_csv(RAW_PATH / "telco_churn.csv")
    required = ['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


def test_churn_distribution():
    """Churn rate should be ~26.5%"""
    df = pd.read_csv(RAW_PATH / "telco_churn.csv")
    churn_rate = (df['Churn'] == 'Yes').mean()
    assert 0.24 <= churn_rate <= 0.29, f"Unexpected churn rate: {churn_rate}"


def test_processed_data_exists():
    """Processed data files should exist"""
    files = ['X_train.npy', 'X_test.npy', 'y_train.npy', 'y_test.npy']
    for f in files:
        assert (PROCESSED_PATH / f).exists(), f"Missing: {f}"


def test_processed_shapes():
    """Processed data should have correct shapes"""
    X_train = np.load(PROCESSED_PATH / "X_train.npy")
    X_test = np.load(PROCESSED_PATH / "X_test.npy")
    y_train = np.load(PROCESSED_PATH / "y_train.npy")
    y_test = np.load(PROCESSED_PATH / "y_test.npy")

    assert X_train.shape[1] == X_test.shape[1], "Feature mismatch"
    assert len(X_train) == len(y_train), "Train X/y mismatch"
    assert len(X_test) == len(y_test), "Test X/y mismatch"
    assert X_train.shape[0] > 5000, "Too few training samples"


def test_no_missing_values():
    """Processed data should have no NaN"""
    X_train = np.load(PROCESSED_PATH / "X_train.npy")
    X_test = np.load(PROCESSED_PATH / "X_test.npy")
    assert not np.isnan(X_train).any(), "NaN in X_train"
    assert not np.isnan(X_test).any(), "NaN in X_test"


def test_smote_balanced():
    """After SMOTE, classes should be balanced"""
    y_train = np.load(PROCESSED_PATH / "y_train.npy")
    churn_rate = y_train.mean()
    assert 0.45 <= churn_rate <= 0.55, f"SMOTE not balanced: {churn_rate}"