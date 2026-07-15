"""
api/main.py
FastAPI backend exposing the Lending Club loan default model.

Thin wrapper per Section 6.1 — only receives requests, calls predict(),
returns the result. No business logic lives here.
"""

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.predict import predict

app = FastAPI(
    title="Loan Default Prediction API",
    description="Predicts whether a loan will be repaid or default, based on Lending Club data.",
    version="1.0.0",
)
@app.get("/health")
def health_check():
    return {"status": "ok"}

class LoanApplication(BaseModel):
    loan_amnt: Optional[float] = None
    term: Optional[str] = None
    int_rate: Optional[float] = None
    installment: Optional[float] = None
    emp_length: Optional[str] = None
    home_ownership: Optional[str] = None
    annual_inc: Optional[float] = None
    verification_status: Optional[str] = None
    pymnt_plan: Optional[str] = None
    purpose: Optional[str] = None
    zip_code: Optional[str] = None
    addr_state: Optional[str] = None
    dti: Optional[float] = None
    delinq_2yrs: Optional[float] = None
    fico_range_low: Optional[float] = None
    fico_range_high: Optional[float] = None
    inq_last_6mths: Optional[float] = None
    open_acc: Optional[float] = None
    pub_rec: Optional[float] = None
    revol_bal: Optional[float] = None
    revol_util: Optional[float] = None
    total_acc: Optional[float] = None
    initial_list_status: Optional[str] = None
    collections_12_mths_ex_med: Optional[float] = None
    application_type: Optional[str] = None
    acc_now_delinq: Optional[float] = None
    tot_coll_amt: Optional[float] = None
    tot_cur_bal: Optional[float] = None
    total_rev_hi_lim: Optional[float] = None
    acc_open_past_24mths: Optional[float] = None
    avg_cur_bal: Optional[float] = None
    bc_open_to_buy: Optional[float] = None
    bc_util: Optional[float] = None
    chargeoff_within_12_mths: Optional[float] = None
    delinq_amnt: Optional[float] = None
    mo_sin_old_il_acct: Optional[float] = None
    mo_sin_old_rev_tl_op: Optional[float] = None
    mo_sin_rcnt_rev_tl_op: Optional[float] = None
    mo_sin_rcnt_tl: Optional[float] = None
    mort_acc: Optional[float] = None
    mths_since_recent_bc: Optional[float] = None
    mths_since_recent_inq: Optional[float] = None
    num_accts_ever_120_pd: Optional[float] = None
    num_actv_bc_tl: Optional[float] = None
    num_actv_rev_tl: Optional[float] = None
    num_bc_sats: Optional[float] = None
    num_bc_tl: Optional[float] = None
    num_il_tl: Optional[float] = None
    num_op_rev_tl: Optional[float] = None
    num_rev_accts: Optional[float] = None
    num_rev_tl_bal_gt_0: Optional[float] = None
    num_sats: Optional[float] = None
    num_tl_120dpd_2m: Optional[float] = None
    num_tl_30dpd: Optional[float] = None
    num_tl_90g_dpd_24m: Optional[float] = None
    num_tl_op_past_12m: Optional[float] = None
    pct_tl_nvr_dlq: Optional[float] = None
    percent_bc_gt_75: Optional[float] = None
    pub_rec_bankruptcies: Optional[float] = None
    tax_liens: Optional[float] = None
    tot_hi_cred_lim: Optional[float] = None
    total_bal_ex_mort: Optional[float] = None
    total_bc_limit: Optional[float] = None
    total_il_high_credit_limit: Optional[float] = None
    disbursement_method: Optional[str] = None
    # Note: earliest_cr_line intentionally excluded — engineer_features()
    # converts it to credit_history_months internally. If you want the
    # API to accept a raw date instead, this schema would need updating.
    credit_history_months: Optional[float] = None


class PredictionResponse(BaseModel):
    prediction: int
    probability_good: float


@app.get("/")
def root():
    return {"message": "Loan Default Prediction API is running. See /docs for usage."}


@app.post("/predict", response_model=PredictionResponse)
def predict_loan_default(application: LoanApplication):
    """
    Predicts loan default risk for a single application.

    Returns:
        prediction: 1 (good/repaid) or 0 (bad/defaulted)
        probability_good: model's confidence the loan will be repaid
    """
    try:
        input_dict = application.model_dump(exclude_none=True)
        result = predict(input_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))