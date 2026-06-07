"""
Data Ingestion Module
Loads and validates raw Telco Churn dataset
Author: Yuvraaj M N
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
RAW_DATA_PATH = Path("data/raw/telco_churn.csv")
PROCESSED_DATA_PATH = Path("data/processed/")
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw dataset"""
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def basic_info(df: pd.DataFrame) -> dict:
    """Print basic dataset info"""
    info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "churn_distribution": df["Churn"].value_counts().to_dict()
    }

    print("\n" + "="*60)
    print("📊 DATASET OVERVIEW")
    print("="*60)
    print(f"Shape: {info['shape']}")
    print(f"\nChurn Distribution:")
    total = sum(info['churn_distribution'].values())
    for k, v in info['churn_distribution'].items():
        print(f"  {k}: {v} ({v/total*100:.1f}%)")

    print(f"\nMissing Values:")
    missing = {k: v for k, v in info['missing'].items() if v > 0}
    if missing:
        for k, v in missing.items():
            print(f"  {k}: {v}")
    else:
        print("  None found")

    return info


def validate_raw_data(df: pd.DataFrame) -> bool:
    """Basic data validation"""
    required_columns = [
        'customerID', 'gender', 'SeniorCitizen', 'Partner',
        'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
    ]

    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        logger.error(f"Missing columns: {missing_cols}")
        return False

    logger.info("✅ Data validation passed")
    return True


if __name__ == "__main__":
    df = load_raw_data()
    basic_info(df)
    validate_raw_data(df)