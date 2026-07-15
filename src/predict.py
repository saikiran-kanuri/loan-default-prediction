"""
predict.py
Loads the trained XGBoost pipeline and exposes a simple predict() function
for the Lending Club model.
Handles partial input: any column not provided by the caller is filled
with NaN before reaching the pipeline, so SimpleImputer can handle it —
the pipeline requires all expected columns to be PRESENT (even if empty),
not just the ones with known values.
"""
import json
import numpy as np
import joblib
import pandas as pd
from src.config import PROJECT_ROOT, TARGET_COLUMN
from src.features import engineer_features
from src.data_loader import load_data, clean_data

MODEL_PATH = PROJECT_ROOT / "models" / "model_v2_xgboost.pkl"
_pipeline = joblib.load(MODEL_PATH)

# The exact set of columns the pipeline was trained on (post-cleaning,
# pre-feature-engineering). Loaded from a small committed JSON file
# instead of the raw CSV, so the API doesn't need the 80MB dataset
# present at runtime — only the trained model and this column list.
EXPECTED_COLUMNS_PATH = PROJECT_ROOT / "src" / "expected_columns.json"
with open(EXPECTED_COLUMNS_PATH) as f:
    EXPECTED_COLUMNS = json.load(f)


def predict(input_dict: dict) -> dict:
    """
    Predicts loan default risk for a single applicant.

    Args:
        input_dict: dictionary of feature_name -> value. Missing keys
            are filled with NaN and handled by the pipeline's imputer.

    Returns:
        dict with:
            - prediction: 1 (good/repaid) or 0 (bad/defaulted)
            - probability_good: model's confidence the loan will be repaid
    """
    # Ensure every expected column is present, even if not provided
    complete_input = {col: input_dict.get(col, np.nan) for col in EXPECTED_COLUMNS}

    input_df = pd.DataFrame([complete_input])
    input_df = engineer_features(input_df)

    prediction = _pipeline.predict(input_df)[0]
    probability_good = _pipeline.predict_proba(input_df)[0][1]

    return {
        "prediction": int(prediction),
        "probability_good": round(float(probability_good), 4),
    }


if __name__ == "__main__":
    # Quick manual test: python3 -m src.predict
    df = load_data()
    df = clean_data(df)

    sample_row = df.drop(columns=[TARGET_COLUMN]).iloc[0].to_dict()
    actual_label = df.iloc[0][TARGET_COLUMN]

    result = predict(sample_row)
    print(f"Actual label: {actual_label}")
    print(f"Prediction result: {result}")