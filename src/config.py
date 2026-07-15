"""
config.py
Central place for file paths, constants, and shared settings.
All other modules import from here — never hardcode paths elsewhere.

Project: Lending Club loan default prediction (Phase 2 dataset).
"""

from pathlib import Path

# --- Project root & paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "lending_club_sample.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "lending_club_clean.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model_v2_full.pkl"

# --- Target construction ---
# loan_status values that represent a resolved outcome.
# 'Current', 'Late', 'In Grace Period' are excluded — unresolved, no label yet.
GOOD_STATUSES = ["Fully Paid", "Does not meet the credit policy. Status:Fully Paid"]
BAD_STATUSES = [
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
]
TARGET_COLUMN = "target"

# --- Columns dropped for missing values (>50% missing) ---
MISSING_VALUE_THRESHOLD = 50  # percent

# --- Columns dropped for data leakage or LC-internal assignment ---
LEAKAGE_COLUMNS = [
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt", "last_credit_pull_d",
    "out_prncp", "out_prncp_inv", "debt_settlement_flag", "issue_d",
    "funded_amnt", "funded_amnt_inv", "loan_status", "title",
    "last_fico_range_high", "last_fico_range_low", "hardship_flag",
]

# Columns that are LC's own internal risk output, not applicant-provided
LC_INTERNAL_COLUMNS = [ "grade", "sub_grade", "id", "url", "emp_title"]

# Constant / zero-variance columns confirmed during EDA
CONSTANT_COLUMNS = ["policy_code"]

# --- Categorical columns (need encoding in train.py's Pipeline) ---
CATEGORICAL_COLUMNS = [
    "term",
    "emp_length",
    "home_ownership",
    "verification_status",
    "pymnt_plan",
    "purpose",
    "zip_code",
    "addr_state",
    "initial_list_status",
    "application_type",
    "disbursement_method",
    
]

# --- Modeling constants ---
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # of the remaining train set, for the 3-way split discussed earlier