"""
features.py
Feature engineering for the Lending Club dataset.

Currently a pass-through — no new engineered features yet, consistent
with the "get the baseline working first" decision from Phase 1.

Missing value imputation is NOT done here — it's handled inside the
sklearn Pipeline in train.py (SimpleImputer), so training-time and
inference-time imputation can never drift out of sync (Section 3.2).

Categorical encoding (one-hot) also happens inside the Pipeline, not here.
"""

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies feature engineering to the cleaned DataFrame.

    - Converts 'earliest_cr_line' into 'credit_history_months'
    - Adds 'debt_to_income_calculated': loan_amnt / annual_inc
      (loan size relative to income — not the same as LC's own 'dti',
      which is based on total monthly debt payments, not loan amount)
    - Adds 'credit_utilization': revol_bal / total_rev_hi_lim
      (standard credit risk metric — % of available revolving credit in use)
    """
    df = df.copy()

    if "earliest_cr_line" in df.columns:
        earliest = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
        reference_date = earliest.max()
        df["credit_history_months"] = (
            (reference_date.year - earliest.dt.year) * 12
            + (reference_date.month - earliest.dt.month)
        )
        df = df.drop(columns=["earliest_cr_line"])

    if "loan_amnt" in df.columns and "annual_inc" in df.columns:
        # avoid division by zero: replace 0 income with NaN,
        # SimpleImputer in train.py will handle the resulting NaN
        safe_income = df["annual_inc"].replace(0, np.nan)
        df["debt_to_income_calculated"] = df["loan_amnt"] / safe_income

    if "revol_bal" in df.columns and "total_rev_hi_lim" in df.columns:
        safe_limit = df["total_rev_hi_lim"].replace(0, np.nan)
        df["credit_utilization"] = df["revol_bal"] / safe_limit

    return df


if __name__ == "__main__":
    # Quick manual test: python3 -m src.features
    from src.data_loader import load_data, clean_data

    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)
    print(df.shape)
    print(df.columns.tolist())