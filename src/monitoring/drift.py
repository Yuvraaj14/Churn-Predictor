"""
Evidently AI Data Drift + Model Monitoring
Generates HTML reports for GitHub Pages
Author: Yuvraaj M N
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime

from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    ClassificationPreset,
    DataQualityPreset
)
'''
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesSummaryMetric,
    ColumnDriftMetric
)
'''

PROCESSED_PATH = Path("data/processed")
MODELS_PATH = Path("models")
REPORTS_PATH = Path("reports")
REPORTS_PATH.mkdir(exist_ok=True)

# Key features to monitor
KEY_FEATURES = [
    'tenure', 'MonthlyCharges', 'TotalCharges',
    'contract_risk', 'num_services', 'avg_monthly_vs_total'
]


def load_reference_data() -> pd.DataFrame:
    """Load training data as reference"""
    df = pd.read_csv(PROCESSED_PATH / "X_train_raw.csv")
    return df


def load_current_data() -> pd.DataFrame:
    """
    Load current/production data.
    In production this would be recent inference data.
    For demo, we use test set as 'current production data'.
    """
    df = pd.read_csv(PROCESSED_PATH / "X_test_raw.csv")
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features for monitoring"""

    df = df.copy()

    # Contract risk
    contract_risk = {'Month-to-month': 2, 'One year': 1, 'Two year': 0}
    df['contract_risk'] = df['Contract'].map(contract_risk).fillna(0)

    # Avg monthly vs total
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
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

    return df


def generate_data_drift_report(reference: pd.DataFrame,
                                current: pd.DataFrame) -> str:
    """Generate data drift report"""

    print("📊 Generating Data Drift Report...")

    # Select numeric + key features
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges',
                   'contract_risk', 'num_services', 'avg_monthly_vs_total']

    ref_subset = reference[numeric_cols].fillna(0)
    cur_subset = current[numeric_cols].fillna(0)

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])

    report.run(
        reference_data=ref_subset,
        current_data=cur_subset
    )

    # Save HTML
    html_path = REPORTS_PATH / "data_drift_report.html"
    report.save_html(str(html_path))
    print(f"   ✅ Data drift report: {html_path}")

    # Extract drift summary
    report_dict = report.as_dict()
    drift_detected = False

    try:
        for metric in report_dict.get('metrics', []):
            if 'DatasetDriftMetric' in str(metric.get('metric', '')):
                result = metric.get('result', {})
                drift_detected = result.get('dataset_drift', False)
                drift_share = result.get('share_of_drifted_columns', 0)
                print(f"   Drift detected: {drift_detected}")
                print(f"   Drifted columns: {drift_share*100:.1f}%")
                break
    except Exception:
        pass

    return str(html_path)


def generate_model_performance_report(reference: pd.DataFrame,
                                       current: pd.DataFrame) -> str:
    """Generate model performance monitoring report"""

    print("\n📊 Generating Model Performance Report...")

    # Load model and get predictions
    model = joblib.load(MODELS_PATH / "xgboost.pkl")
    preprocessor = joblib.load(PROCESSED_PATH / "preprocessor.pkl")

    # Load target
    y_train = pd.read_csv(PROCESSED_PATH / "y_train_raw.csv").iloc[:, 0]
    y_test = pd.read_csv(PROCESSED_PATH / "y_test_raw.csv").iloc[:, 0]

    # Get predictions for reference (train sample)
    ref_with_features = add_engineered_features(reference)
    cur_with_features = add_engineered_features(current)

    # Add tenure_group
    for df in [ref_with_features, cur_with_features]:
        df['tenure_group'] = pd.cut(
            df['tenure'],
            bins=[0, 12, 24, 48, 60, 72],
            labels=['0-1yr', '1-2yr', '2-4yr', '4-5yr', '5-6yr']
        )

    # Sample reference to match test size
    ref_sample = ref_with_features.sample(
        min(len(current), len(reference)),
        random_state=42
    )
    y_ref_sample = y_train.iloc[ref_sample.index % len(y_train)].values

    # Preprocess
    X_ref = preprocessor.transform(ref_sample)
    X_cur = preprocessor.transform(cur_with_features)

    # Predictions
    ref_preds = model.predict(X_ref)
    ref_probs = model.predict_proba(X_ref)[:, 1]
    cur_preds = model.predict(X_cur)
    cur_probs = model.predict_proba(X_cur)[:, 1]

    # Build evidently dataframes
    ref_df = pd.DataFrame({
        'target': y_ref_sample[:len(ref_preds)],
        'prediction': ref_preds,
        'prediction_prob': ref_probs
    })

    cur_df = pd.DataFrame({
        'target': y_test.values[:len(cur_preds)],
        'prediction': cur_preds,
        'prediction_prob': cur_probs
    })

    report = Report(metrics=[ClassificationPreset()])

    report.run(
        reference_data=ref_df,
        current_data=cur_df,
        column_mapping=None
    )

    html_path = REPORTS_PATH / "model_performance_report.html"
    report.save_html(str(html_path))
    print(f"   ✅ Model performance report: {html_path}")

    return str(html_path)


def generate_summary_json() -> dict:
    """Generate summary metrics JSON for dashboard"""

    summary = {
        "generated_at": datetime.now().isoformat(),
        "model": "XGBoost v2.0",
        "dataset": "Telco Customer Churn",
        "metrics": {
            "train_samples": 8278,
            "test_samples": 1409,
            "churn_rate_train": "50.0% (after SMOTE)",
            "churn_rate_test": "26.5%"
        },
        "model_performance": {
            "accuracy": 0.7438,
            "roc_auc": 0.8416,
            "f1": 0.6297,
            "precision": 0.5108,
            "recall": 0.8209
        },
        "top_churn_features": [
            "Contract_Month-to-month",
            "contract_risk",
            "avg_monthly_vs_total",
            "OnlineSecurity_No",
            "InternetService_Fiber optic",
            "TechSupport_No",
            "PaymentMethod_Electronic check",
            "tenure",
            "MonthlyCharges",
            "Dependents_No"
        ],
        "ab_test": {
            "champion": "XGBoost (80% traffic)",
            "challenger": "LightGBM (20% traffic)",
            "baseline": "Logistic Regression (reference)"
        },
        "cache": {
            "backend": "Redis",
            "ttl_seconds": 3600,
            "target_hit_rate": "40%"
        }
    }

    summary_path = REPORTS_PATH / "monitoring_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Summary JSON: {summary_path}")
    return summary


def run_all_monitoring():
    """Run complete monitoring pipeline"""

    print("\n" + "="*60)
    print("🔍 EVIDENTLY AI MONITORING PIPELINE")
    print("="*60)

    # Load data
    print("\n📂 Loading data...")
    reference = load_reference_data()
    current = load_current_data()

    reference = add_engineered_features(reference)
    current = add_engineered_features(current)

    print(f"   Reference: {len(reference)} samples (training data)")
    print(f"   Current:   {len(current)} samples (test/production data)")

    # Generate reports
    drift_path = generate_data_drift_report(reference, current)
    perf_path = generate_model_performance_report(reference, current)
    summary = generate_summary_json()

    print("\n" + "="*60)
    print("✅ MONITORING COMPLETE!")
    print("="*60)
    print(f"   Data Drift Report:  {drift_path}")
    print(f"   Performance Report: {perf_path}")
    print(f"   Summary JSON:       reports/monitoring_summary.json")
    print("\n💡 Next: Deploy these reports to GitHub Pages!")


if __name__ == "__main__":
    run_all_monitoring()