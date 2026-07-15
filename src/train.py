"""
train.py
Trains and compares Logistic Regression and XGBoost on the Lending
Club dataset, using an identical Pipeline structure and identical
train/validation/test splits for a fair comparison (Section 4.1, 6.2).

Run with: python3 -m src.train --model logistic
          python3 -m src.train --model xgboost
"""

import argparse
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

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


def build_preprocessor(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    """Shared preprocessing — identical for both models, for a fair comparison."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def build_pipeline(model_type: str, numeric_cols: list, categorical_cols: list) -> Pipeline:
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    if model_type == "logistic":
        model = LogisticRegression(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=1000,
        )
    elif model_type == "xgboost":
        # scale_pos_weight handles class imbalance for XGBoost —
        # the XGBoost equivalent of class_weight='balanced'.
        # Ratio = (# negative class) / (# positive class)
        model = XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="auc",
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def main(model_type: str):
    # --- Load and prepare data (identical for both models) ---
    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    numeric_cols = [c for c in X.columns if c not in CATEGORICAL_COLUMNS]
    categorical_cols = [c for c in CATEGORICAL_COLUMNS if c in X.columns]

    # --- Three-way stratified split — SAME random_state as before,
    # guarantees identical train/val/test rows for both models ---
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    val_ratio_of_temp = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_of_temp, random_state=RANDOM_STATE, stratify=y_temp,
    )

    print(f"Model: {model_type}")
    print(f"Train: {X_train.shape[0]} | Validation: {X_val.shape[0]} | Test: {X_test.shape[0]}\n")

    # --- Handle XGBoost's scale_pos_weight (needs the actual training ratio) ---
    pipeline = build_pipeline(model_type, numeric_cols, categorical_cols)
    if model_type == "xgboost":
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        pipeline.named_steps["model"].set_params(scale_pos_weight=neg / pos)

    pipeline.fit(X_train, y_train)

    # --- Evaluate on validation set ---
    y_val_pred = pipeline.predict(X_val)
    y_val_proba = pipeline.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val, y_val_proba)
    print(f"Validation AUC-ROC: {auc:.4f}\n")
    print("Validation Classification Report:")
    print(classification_report(y_val, y_val_pred))
    print("Validation Confusion Matrix:")
    print(confusion_matrix(y_val, y_val_pred))

    # --- Save with a model-specific filename ---
    model_path = PROJECT_ROOT / "models" / f"model_v2_{model_type}.pkl"
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["logistic", "xgboost"], default="logistic")
    args = parser.parse_args()
    main(args.model)