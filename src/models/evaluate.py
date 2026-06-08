"""
Model Evaluation + SHAP Feature Importance
Author: Yuvraaj M N
"""

import numpy as np
import pandas as pd
import shap
import joblib
import json
import matplotlib.pyplot as plt
from pathlib import Path

PROCESSED_PATH = Path("data/processed")
MODELS_PATH = Path("models")
REPORTS_PATH = Path("reports")
REPORTS_PATH.mkdir(exist_ok=True)


def load_model_and_data(model_name="xgboost"):
    """Load model and test data"""
    model_path = MODELS_PATH / f"{model_name}.pkl"
    model = joblib.load(model_path)

    X_test = np.load(PROCESSED_PATH / "X_test.npy")
    y_test = np.load(PROCESSED_PATH / "y_test.npy")

    # Load feature names from preprocessor
    preprocessor = joblib.load(PROCESSED_PATH / "preprocessor.pkl")
    feature_names = []

    # Get feature names from each transformer
    for name, transformer, cols in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(cols)
        elif name == 'cat':
            ohe = transformer.named_steps['onehot']
            feature_names.extend(ohe.get_feature_names_out(cols).tolist())
        elif name == 'bin':
            feature_names.extend(cols)

    return model, X_test, y_test, feature_names


def generate_shap_plots(model, X_test, feature_names, model_name="XGBoost"):
    """Generate SHAP visualizations"""

    print(f"\n🔍 Generating SHAP explanations for {model_name}...")

    # Use sample for speed
    sample_size = min(500, len(X_test))
    X_sample = X_test[:sample_size]

    # Create SHAP explainer
    if "xgboost" in model_name.lower() or "lightgbm" in model_name.lower():
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, X_sample)

    shap_values = explainer.shap_values(X_sample)

    # Handle binary classification output
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # ─── Plot 1: Summary Bar ───
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        plot_type="bar",
        max_display=15,
        show=False
    )
    plt.title(f'SHAP Feature Importance — {model_name}', fontsize=14, pad=20)
    plt.tight_layout()
    bar_path = REPORTS_PATH / f"shap_bar_{model_name.lower()}.png"
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ SHAP bar chart saved: {bar_path}")

    # ─── Plot 2: Beeswarm ───
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        max_display=15,
        show=False
    )
    plt.title(f'SHAP Summary (Beeswarm) — {model_name}', fontsize=14, pad=20)
    plt.tight_layout()
    beeswarm_path = REPORTS_PATH / f"shap_beeswarm_{model_name.lower()}.png"
    plt.savefig(beeswarm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ SHAP beeswarm saved: {beeswarm_path}")

    # ─── Top features ───
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_features = pd.Series(mean_shap, index=feature_names)\
        .sort_values(ascending=False).head(10)

    print(f"\n📊 Top 10 Most Important Features ({model_name}):")
    for feat, val in top_features.items():
        bar = "█" * int(val * 100)
        print(f"   {feat:<35} {bar} {val:.4f}")

    # Save top features
    top_features.to_csv(REPORTS_PATH / f"top_features_{model_name.lower()}.csv")

    return shap_values, top_features


def run_full_evaluation():
    """Run complete model evaluation"""

    print("\n" + "="*60)
    print("📊 MODEL EVALUATION + SHAP ANALYSIS")
    print("="*60)

    # Load best model info
    with open(MODELS_PATH / "best_model_info.json") as f:
        best_info = json.load(f)

    champion = best_info["champion"].lower().replace(" ", "_")
    challenger = best_info["challenger"].lower().replace(" ", "_")

    # Evaluate champion
    print(f"\n🏆 Champion: {best_info['champion']}")
    model, X_test, y_test, feature_names = load_model_and_data(champion)
    shap_vals, top_features = generate_shap_plots(
        model, X_test, feature_names, best_info["champion"]
    )

    # Evaluate challenger
    print(f"\n🥊 Challenger: {best_info['challenger']}")
    model_c, X_test_c, y_test_c, feature_names_c = load_model_and_data(challenger)
    shap_vals_c, top_features_c = generate_shap_plots(
        model_c, X_test_c, feature_names_c, best_info["challenger"]
    )

    print("\n✅ Evaluation complete!")
    print(f"   Reports saved to: {REPORTS_PATH}/")


if __name__ == "__main__":
    run_full_evaluation()