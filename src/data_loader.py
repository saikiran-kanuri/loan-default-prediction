"""
data_loader.py
Loads and performs cleaning on the Lending Club sampled dataset.

Cleaning steps (decided during EDA):
1. Filter to resolved loans only, build binary target from loan_status
2. Drop columns with >50% missing values
3. Drop known leakage columns
4. Drop LC-internal / redundant columns
5. Drop constant (zero-variance) columns
"""

import pandas as pd

from src.config import (
    RAW_DATA_PATH,
    GOOD_STATUSES,
    BAD_STATUSES,
    TARGET_COLUMN,
    MISSING_VALUE_THRESHOLD,
    LEAKAGE_COLUMNS,
    LC_INTERNAL_COLUMNS,
    CONSTANT_COLUMNS,
)


def load_data(filepath=None) -> pd.DataFrame:
    """Loads the raw sampled Lending Club CSV into a DataFrame."""
    path = filepath or RAW_DATA_PATH
    df = pd.read_csv(path, low_memory=False)
    return df


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters to resolved loans only and builds the binary target column.

    Why: 'Current', 'Late', 'In Grace Period' loans have no resolved
    outcome yet — including them would mislabel unresolved loans as
    'good', corrupting the target variable.
    """
    df = df.copy()
    resolved_statuses = GOOD_STATUSES + BAD_STATUSES
    df = df[df["loan_status"].isin(resolved_statuses)].copy()
    df[TARGET_COLUMN] = df["loan_status"].apply(
        lambda x: 1 if x in GOOD_STATUSES else 0
    )
    return df


def drop_unusable_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns in three categories, decided during EDA:
    - >50% missing values
    - known data leakage columns
    - LC-internal / redundant / constant columns
    """
    df = df.copy()

    # 1. High-missing columns
    missing_pct = df.isnull().mean() * 100
    high_missing_cols = missing_pct[missing_pct > MISSING_VALUE_THRESHOLD].index.tolist()

    # 2. Combine all drop lists, keep only columns actually present
    all_drop_cols = set(high_missing_cols) | set(LEAKAGE_COLUMNS) | set(LC_INTERNAL_COLUMNS) | set(CONSTANT_COLUMNS)
    cols_to_drop = [c for c in all_drop_cols if c in df.columns]

    df = df.drop(columns=cols_to_drop)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline: build target, then drop unusable columns.
    Note: loan_status is dropped inside drop_unusable_columns (it's in
    LEAKAGE_COLUMNS), so target must be built BEFORE dropping columns.
    """
    df = build_target(df)
    df = drop_unusable_columns(df)
    return df


if __name__ == "__main__":
    # Quick manual test: python3 -m src.data_loader
    df = load_data()
    df = clean_data(df)
    print(df.shape)
    print(df[TARGET_COLUMN].value_counts(normalize=True))