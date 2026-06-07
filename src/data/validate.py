"""
Data Validation
Author: Yuvraaj M N
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def validate_schema(df: pd.DataFrame) -> bool:
    """Validate column schema"""
    required = [
        'customerID', 'gender', 'SeniorCitizen', 'Partner',
        'Dependents', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn'
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error(f"Missing columns: {missing}")
        return False
    return True


def validate_ranges(df: pd.DataFrame) -> bool:
    """Validate value ranges"""
    checks = [
        ("tenure >= 0", df['tenure'].min() >= 0),
        ("MonthlyCharges > 0", df['MonthlyCharges'].min() > 0),
        ("SeniorCitizen in [0,1]", df['SeniorCitizen'].isin([0, 1]).all()),
        ("Churn in [Yes,No]", df['Churn'].isin(['Yes', 'No']).all()),
    ]

    all_passed = True
    for name, result in checks:
        status = "✅" if result else "❌"
        print(f"  {status} {name}")
        if not result:
            all_passed = False

    return all_passed


def validate_no_duplicates(df: pd.DataFrame) -> bool:
    """Check for duplicate customerIDs"""
    dupes = df['customerID'].duplicated().sum()
    if dupes > 0:
        logger.warning(f"Found {dupes} duplicate customerIDs")
        return False
    return True


def run_all_validations(df: pd.DataFrame) -> bool:
    """Run all validation checks"""
    print("\n" + "="*60)
    print("🔍 DATA VALIDATION")
    print("="*60)

    results = {
        "Schema": validate_schema(df),
        "Ranges": validate_ranges(df),
        "No Duplicates": validate_no_duplicates(df)
    }

    print("\n📋 Summary:")
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check}: {status}")

    all_passed = all(results.values())
    print(f"\n{'✅ ALL VALIDATIONS PASSED' if all_passed else '❌ SOME VALIDATIONS FAILED'}")
    return all_passed


if __name__ == "__main__":
    from src.data.ingest import load_raw_data
    df = load_raw_data()
    run_all_validations(df)