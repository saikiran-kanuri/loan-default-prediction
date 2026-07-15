"""
evaluate.py
Generates SHAP explainability plots for the trained XGBoost pipeline.

Run with: python3 -m src.evaluate
"""

import joblib
import matplotlib.pyplot as plt
import shap

from src.config import (
    CATEGORICAL_COLUMNS,
    TARGET_COLUMN,
    PROJECT_ROOT,
    RANDOM_STATE,
    TEST_SIZE,
    VAL_SIZE,
)
from src.data_loader import load_data, clean_data
from src.features import engineer_features
from sklearn.model_selection import train_test_split


def main():
    # --- Load the same data + splits used in training, for consistency ---
    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    val_ratio_of_temp = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_of_temp, random_state=RANDOM_STATE, stratify=y_temp,
    )

    # --- Load the saved XGBoost pipeline ---
    model_path = PROJECT_ROOT / "models" / "model_v2_xgboost.pkl"
    pipeline = joblib.load(model_path)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    # --- Transform validation data through preprocessing ---
    # SHAP needs already-encoded numeric data, not raw categorical strings.
    # Use a sample for speed — SHAP on the full validation set can be slow.
    X_val_sample = X_val.sample(n=min(2000, len(X_val)), random_state=RANDOM_STATE)
    X_val_transformed = preprocessor.transform(X_val_sample)

    # Get readable feature names after one-hot encoding
    feature_names = preprocessor.get_feature_names_out()

    # --- Compute SHAP values ---
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val_transformed)

    # --- Summary plot: overall feature importance ---
    plt.figure()
    shap.summary_plot(
        shap_values, X_val_transformed, feature_names=feature_names, show=False
    )
    plt.tight_layout()
    summary_path = PROJECT_ROOT / "models" / "shap_summary_plot.png"
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    print(f"SHAP summary plot saved to: {summary_path}")

    # --- Bar plot: mean absolute SHAP value per feature (cleaner ranking) ---
    plt.figure()
    shap.summary_plot(
        shap_values, X_val_transformed, feature_names=feature_names,
        plot_type="bar", show=False
    )
    plt.tight_layout()
    bar_path = PROJECT_ROOT / "models" / "shap_bar_plot.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    print(f"SHAP bar plot saved to: {bar_path}")


if __name__ == "__main__":
    main()