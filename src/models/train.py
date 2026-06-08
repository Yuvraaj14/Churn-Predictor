"""
Model Training with MLflow + DagsHub Tracking
Trains XGBoost, LightGBM, and Logistic Regression
Author: Yuvraaj M N
"""

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import os
import joblib
import json
from pathlib import Path
from dotenv import load_dotenv

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, classification_report,
    confusion_matrix
)
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()

# ============================================
# PATHS
# ============================================
PROCESSED_PATH = Path("data/processed")
MODELS_PATH = Path("models")
MODELS_PATH.mkdir(exist_ok=True)
REPORTS_PATH = Path("reports")
REPORTS_PATH.mkdir(exist_ok=True)

# ============================================
# MLFLOW SETUP
# ============================================
def setup_mlflow():
    """Configure MLflow with DagsHub"""

    dagshub_user = os.getenv("DAGSHUB_USER_NAME", "Yuvraaj14")
    dagshub_token = os.getenv("DAGSHUB_TOKEN", "")
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        f"https://dagshub.com/{dagshub_user}/churn-predictor.mlflow"
    )

    # Set credentials
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("churn-prediction")

    print(f"✅ MLflow configured")
    print(f"   Tracking URI: {tracking_uri}")
    print(f"   Experiment: churn-prediction")


# ============================================
# LOAD DATA
# ============================================
def load_processed_data():
    """Load preprocessed numpy arrays"""

    X_train = np.load(PROCESSED_PATH / "X_train.npy")
    X_test = np.load(PROCESSED_PATH / "X_test.npy")
    y_train = np.load(PROCESSED_PATH / "y_train.npy")
    y_test = np.load(PROCESSED_PATH / "y_test.npy")

    print(f"\n📂 Data loaded:")
    print(f"   X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"   X_test:  {X_test.shape}  | y_test:  {y_test.shape}")

    return X_train, X_test, y_train, y_test


# ============================================
# METRICS
# ============================================
def compute_metrics(y_true, y_pred, y_prob):
    """Compute all evaluation metrics"""
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
    }


# ============================================
# CONFUSION MATRIX PLOT
# ============================================
def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """Save confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['No Churn', 'Churn'],
        yticklabels=['No Churn', 'Churn']
    )
    plt.title(f'Confusion Matrix — {model_name}', fontsize=14)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Confusion matrix saved: {save_path}")


# ============================================
# TRAIN MODELS
# ============================================
def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression baseline"""
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    """Train XGBoost with better hyperparameters"""
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=2.77,  # handles class imbalance
        eval_metric='auc',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def train_lightgbm(X_train, y_train):
    """Train LightGBM with better hyperparameters"""
    model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(X_train, y_train)
    return model


# ============================================
# RUN EXPERIMENT
# ============================================
def run_experiment(model_name, model, X_train, X_test,
                   y_train, y_test, params):
    """Run single MLflow experiment"""

    print(f"\n{'='*60}")
    print(f"🚀 Training: {model_name}")
    print(f"{'='*60}")

    with mlflow.start_run(run_name=model_name):

        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("test_samples", X_test.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])

        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        metrics = compute_metrics(y_test, y_pred, y_prob)
        mlflow.log_metrics(metrics)

        # Print metrics
        print(f"\n📊 Metrics:")
        for k, v in metrics.items():
            print(f"   {k}: {v}")

        # Classification report
        report = classification_report(
            y_test, y_pred,
            target_names=['No Churn', 'Churn']
        )
        print(f"\n{report}")

        # Save confusion matrix
        cm_path = REPORTS_PATH / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        plot_confusion_matrix(y_test, y_pred, model_name, cm_path)
        mlflow.log_artifact(str(cm_path))

        # Save model locally
        model_path = MODELS_PATH / f"{model_name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path))

        # Log model to MLflow registry
        if model_name == "XGBoost":
            mlflow.xgboost.log_model(
                model, "model",
                registered_model_name="churn-xgboost"
            )
        elif model_name == "LightGBM":
            mlflow.lightgbm.log_model(
                model, "model",
                registered_model_name="churn-lightgbm"
            )
        else:
            mlflow.sklearn.log_model(
                model, "model",
                registered_model_name="churn-logistic"
            )

        run_id = mlflow.active_run().info.run_id
        print(f"\n✅ Run logged to MLflow: {run_id}")

    return metrics, run_id


# ============================================
# COMPARE MODELS
# ============================================
def compare_models(results: dict):
    """Print model comparison table"""

    print(f"\n{'='*60}")
    print(f"📊 MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'AUC':>8} {'F1':>8} {'Accuracy':>10} {'Precision':>10} {'Recall':>8}")
    print("-"*60)

    best_model = max(results.items(), key=lambda x: x[1]['metrics']['roc_auc'])

    for model_name, result in results.items():
        m = result['metrics']
        marker = " ← CHAMPION" if model_name == best_model[0] else ""
        print(f"{model_name:<20} {m['roc_auc']:>8.4f} {m['f1']:>8.4f} "
              f"{m['accuracy']:>10.4f} {m['precision']:>10.4f} {m['recall']:>8.4f}{marker}")

    print(f"\n🏆 Best Model: {best_model[0]} (AUC: {best_model[1]['metrics']['roc_auc']:.4f})")

    # Save comparison
    comparison = {
        name: {"metrics": res["metrics"], "run_id": res["run_id"]}
        for name, res in results.items()
    }
    with open(REPORTS_PATH / "model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    return best_model[0]


# ============================================
# MAIN
# ============================================
def main():
    print("\n" + "="*60)
    print("🤖 CHURN PREDICTOR — MODEL TRAINING")
    print("="*60)

    # Setup MLflow
    setup_mlflow()

    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()

    results = {}

    # ─── Model 1: Logistic Regression (Baseline) ───
    lr_params = {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"}
    lr_model = train_logistic_regression(X_train, y_train)
    lr_metrics, lr_run_id = run_experiment(
        "Logistic Regression", lr_model,
        X_train, X_test, y_train, y_test, lr_params
    )
    results["Logistic Regression"] = {"metrics": lr_metrics, "run_id": lr_run_id}

    # ─── Model 2: XGBoost (Champion) ───
    xgb_params = {
        "n_estimators": 500, "max_depth": 4,
        "learning_rate": 0.03, "subsample": 0.8,
        "colsample_bytree": 0.7, "min_child_weight": 5,
        "gamma": 0.1, "scale_pos_weight": 2.77
    }
    xgb_model = train_xgboost(X_train, y_train)
    xgb_metrics, xgb_run_id = run_experiment(
        "XGBoost", xgb_model,
        X_train, X_test, y_train, y_test, xgb_params
    )
    results["XGBoost"] = {"metrics": xgb_metrics, "run_id": xgb_run_id}

    # ─── Model 3: LightGBM (Challenger) ───
    lgb_params = {
        "n_estimators": 500, "max_depth": 4,
        "learning_rate": 0.03, "subsample": 0.8,
        "colsample_bytree": 0.7, "min_child_samples": 20,
        "class_weight": "balanced"
    }
    lgb_model = train_lightgbm(X_train, y_train)
    lgb_metrics, lgb_run_id = run_experiment(
        "LightGBM", lgb_model,
        X_train, X_test, y_train, y_test, lgb_params
    )
    results["LightGBM"] = {"metrics": lgb_metrics, "run_id": lgb_run_id}

    # ─── Compare ───
    best_model_name = compare_models(results)

    # ─── Save best model info ───
    best_info = {
        "champion": best_model_name,
        "challenger": "LightGBM" if best_model_name == "XGBoost" else "XGBoost",
        "metrics": results[best_model_name]["metrics"],
        "run_id": results[best_model_name]["run_id"]
    }
    with open(MODELS_PATH / "best_model_info.json", "w") as f:
        json.dump(best_info, f, indent=2)

    print(f"\n✅ Best model info saved")
    print(f"✅ Training complete! Check DagsHub for MLflow dashboard")
    print(f"\n🔗 MLflow Dashboard: https://dagshub.com/Yuvraaj14/churn-predictor")

    return results


if __name__ == "__main__":
    main()