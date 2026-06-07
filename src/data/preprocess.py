"""
Data Preprocessing + Feature Engineering
Author: Yuvraaj M N
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_PATH = Path("data/processed/")
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)


# ============================================
# FEATURE GROUPS
# ============================================
NUMERIC_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']

CATEGORICAL_FEATURES = [
    'gender', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

BINARY_FEATURES = ['SeniorCitizen']

TARGET = 'Churn'


# ============================================
# PREPROCESSING STEPS
# ============================================
def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Fix TotalCharges — has empty strings for new customers"""
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Fill with MonthlyCharges for new customers (tenure=0)
    mask = df['TotalCharges'].isna()
    df.loc[mask, 'TotalCharges'] = df.loc[mask, 'MonthlyCharges']
    logger.info(f"Fixed {mask.sum()} TotalCharges values")
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode target: Yes=1, No=0"""
    df = df.copy()
    df[TARGET] = (df[TARGET] == 'Yes').astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing ones
    WHY: Domain knowledge improves model performance
    """
    df = df.copy()

    # Tenure groups (customer lifecycle stage)
    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 60, 72],
        labels=['0-1yr', '1-2yr', '2-4yr', '4-5yr', '5-6yr']
    )

    # Avg monthly spend vs total
    df['avg_monthly_vs_total'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)

    # Number of services subscribed
    service_cols = [
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    df['num_services'] = df[service_cols].apply(
        lambda row: sum(1 for v in row if v not in ['No', 'No internet service', 'No phone service']),
        axis=1
    )

    # High value customer flag
    df['is_high_value'] = (df['MonthlyCharges'] > df['MonthlyCharges'].median()).astype(int)

    # Long term customer
    df['is_long_term'] = (df['tenure'] > 24).astype(int)

    # Contract risk score
    contract_risk = {'Month-to-month': 2, 'One year': 1, 'Two year': 0}
    df['contract_risk'] = df['Contract'].map(contract_risk)

    logger.info("✅ Feature engineering complete")
    logger.info(f"   New features: tenure_group, avg_monthly_vs_total, num_services, is_high_value, is_long_term, contract_risk")

    return df


# ============================================
# BUILD PIPELINE
# ============================================
def build_preprocessor() -> ColumnTransformer:
    """Build sklearn preprocessing pipeline"""

    # Add engineered numeric features
    numeric_features = NUMERIC_FEATURES + [
        'avg_monthly_vs_total', 'num_services',
        'is_high_value', 'is_long_term', 'contract_risk'
    ]

    categorical_features = CATEGORICAL_FEATURES + ['tenure_group']

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('bin', 'passthrough', BINARY_FEATURES)
    ])

    return preprocessor


# ============================================
# MAIN PREPROCESSING FUNCTION
# ============================================
def preprocess_data(df: pd.DataFrame, test_size: float = 0.2,
                    apply_smote: bool = True):
    """
    Full preprocessing pipeline:
    1. Fix data issues
    2. Encode target
    3. Feature engineering
    4. Train/test split
    5. SMOTE oversampling
    6. Transform features
    """

    print("\n" + "="*60)
    print("🔧 PREPROCESSING PIPELINE")
    print("="*60)

    # Step 1: Basic fixes
    df = fix_total_charges(df)
    df = encode_target(df)
    print(f"✅ Step 1: Basic fixes done")

    # Step 2: Feature engineering
    df = engineer_features(df)
    print(f"✅ Step 2: Feature engineering done")

    # Step 3: Drop customerID
    df = df.drop('customerID', axis=1)

    # Step 4: Split features and target
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    print(f"\n   Total samples: {len(X)}")
    print(f"   Churn rate: {y.mean()*100:.1f}%")

    # Step 5: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    print(f"\n✅ Step 3: Train/test split")
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

    # Step 6: Build and fit preprocessor
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print(f"\n✅ Step 4: Preprocessing applied")
    print(f"   Features after encoding: {X_train_processed.shape[1]}")

    # Step 7: SMOTE (balance classes)
    if apply_smote:
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(
            X_train_processed, y_train
        )
        print(f"\n✅ Step 5: SMOTE applied")
        print(f"   Before: {len(X_train_processed)} samples | Churn: {y_train.mean()*100:.1f}%")
        print(f"   After:  {len(X_train_resampled)} samples | Churn: {y_train_resampled.mean()*100:.1f}%")
    else:
        X_train_resampled = X_train_processed
        y_train_resampled = y_train

    # Step 8: Save preprocessor
    preprocessor_path = PROCESSED_PATH / "preprocessor.pkl"
    joblib.dump(preprocessor, preprocessor_path)
    print(f"\n✅ Preprocessor saved: {preprocessor_path}")

    # Step 9: Save processed data
    np.save(PROCESSED_PATH / "X_train.npy", X_train_resampled)
    np.save(PROCESSED_PATH / "X_test.npy", X_test_processed)
    np.save(PROCESSED_PATH / "y_train.npy", y_train_resampled)
    np.save(PROCESSED_PATH / "y_test.npy", y_test)

    # Save raw splits for drift detection reference
    X_train.to_csv(PROCESSED_PATH / "X_train_raw.csv", index=False)
    X_test.to_csv(PROCESSED_PATH / "X_test_raw.csv", index=False)
    y_train.to_csv(PROCESSED_PATH / "y_train_raw.csv", index=False)
    y_test.to_csv(PROCESSED_PATH / "y_test_raw.csv", index=False)

    print(f"\n✅ Processed data saved to {PROCESSED_PATH}")
    print("="*60)

    return (X_train_resampled, X_test_processed,
            y_train_resampled, y_test,
            preprocessor)


if __name__ == "__main__":
    from src.data.ingest import load_raw_data, basic_info, validate_raw_data

    # Load
    df = load_raw_data()
    basic_info(df)
    validate_raw_data(df)

    # Preprocess
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    print(f"\n🎯 Final shapes:")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_test:  {X_test.shape}")
    print(f"   y_train: {y_train.shape}")
    print(f"   y_test:  {y_test.shape}")